import numpy as np
import torch
from src.data import context, first_day_targets
from src.models import make_policy, policy_logits, append_action, RewardModel
from src.evaluate import rank_metrics
from src.train import leave_one_out, reward_value


def test_context_retains_latest_and_pads_left():
    assert context([3, 4])[-2:].tolist() == [3, 4]
    assert context(list(range(40))).tolist() == list(range(8, 40))


def test_first_day_does_not_invent_order():
    days = np.array(["2021-07-01", "2021-07-12", "2021-07-12", "2021-07-13"], dtype="datetime64[D]")
    assert first_day_targets(
        days, np.array([2, 3, 4, 5]), np.datetime64("2021-07-12"), np.datetime64("2021-07-14")
    ) == [3, 4]


def test_unknown_targets_remain_in_denominator():
    recall, ndcg = rank_metrics([2, 3], [2, -7])
    assert recall == 0.5
    assert np.isclose(ndcg, 1 / (1 + 1 / np.log2(3)))


def test_empty_targets_are_zero():
    assert rank_metrics([2], []) == (0.0, 0.0)


def test_leave_one_out_excludes_own_return():
    r = torch.tensor([1.0, 2.0, 3.0, 4.0, 8.0, 8.0, 8.0, 8.0])
    assert torch.allclose(
        leave_one_out(r), torch.tensor([-2.0, -2 / 3, 2 / 3, 2.0, 0.0, 0.0, 0.0, 0.0])
    )


def test_special_tokens_masked_and_gradients_flow():
    torch.set_num_threads(2)
    torch.manual_seed(42)
    m = make_policy(20)
    x = torch.tensor(np.stack([context([2, 3]), context([4, 5])]))
    out = policy_logits(m, x)
    assert torch.all(out[:, :2] < -1e8)
    torch.nn.functional.cross_entropy(out, torch.tensor([6, 7])).backward()
    assert m.transformer.wte.weight.grad.abs().sum() > 0


def test_append_is_action_not_observed_feedback():
    x = torch.tensor(np.stack([context([3, 4])]))
    out = append_action(x, torch.tensor([9]))
    assert out.shape == x.shape and out[0, -3:].tolist() == [3, 4, 9]


def test_reward_bounded_and_history_conditioned():
    torch.manual_seed(42)
    m = RewardModel(20).eval()
    x = torch.tensor(np.stack([context([2, 3]), context([7, 8])]))
    p = m(x, torch.tensor([9, 9]))
    assert ((p > 0) & (p < 1)).all() and not torch.equal(p[0], p[1])


def test_repeat_penalty():
    mu = torch.tensor([[0.8, 0.5], [0.8, 0.5]])
    sd = torch.zeros_like(mu)
    r = reward_value(mu, sd, torch.tensor([0.0, 1.0]), torch.zeros(2), "conservative")
    assert torch.isclose(r[0] - r[1], torch.tensor(0.3))


def test_group_baseline_matches_exact_bandit_gradient_in_expectation():
    # Enumerate two independent Bernoulli trajectories. Leave-one-out is
    # independent of the current action, so its expected score gradient is zero.
    p = 0.3
    expected = 0.0
    for a in [0, 1]:
        for b in [0, 1]:
            probability = (p if a else 1 - p) * (p if b else 1 - p)
            advantage = a - b
            expected += probability * advantage * (a - p)
    assert np.isclose(expected, p * (1 - p))


def test_return_head_is_not_used_by_completion_reward():
    mu = torch.tensor([[0.6, 0.1], [0.6, 0.9]])
    result = reward_value(mu, torch.zeros_like(mu), torch.zeros(2), torch.zeros(2), "completion")
    assert result[0] == result[1]


def test_padding_values_do_not_affect_valid_policy_logits():
    torch.manual_seed(42)
    m = make_policy(20).eval()
    x = torch.tensor(np.stack([context([3, 4, 5])]))
    with torch.no_grad():
        before = policy_logits(m, x)[:, 2:]
        m.transformer.wte.weight[0].fill_(100)
        after = policy_logits(m, x)[:, 2:]
    assert torch.allclose(before, after, atol=1e-5)
