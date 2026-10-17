"""Build publication figures and a report strictly from saved measurements."""
import json
import os
import tempfile
from pathlib import Path
os.environ.setdefault('MPLCONFIGDIR',str(Path(tempfile.gettempdir())/'kion-rl-mpl'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from .data import ROOT

LABELS={'baseline':'GPT-2','replay_kl_control':'Replay + KL','completion_weak_kl':'Completion · weak KL','completion_strong_kl':'Completion · strong KL','conservative':'Conservative','continuation_mix':'Continuation mix'}
COLORS={'baseline':'#293e55','replay_kl_control':'#999999','completion_weak_kl':'#c35a35','completion_strong_kl':'#d39c30','conservative':'#147d6f','continuation_mix':'#7660a4'}


def table(df,columns):
    lines=['| '+' | '.join(columns)+' |','| '+' | '.join(['---']*len(columns))+' |']
    for _,r in df.iterrows():
        cells=[]
        for c in columns:
            value=r[c]
            cells.append(f'{value:.4f}' if isinstance(value,(float,np.floating)) else str(value))
        lines.append('| '+' | '.join(cells)+' |')
    return '\n'.join(lines)


def main():
    out=ROOT/'results';reports=ROOT/'reports';reports.mkdir(exist_ok=True)
    metrics=pd.read_csv(out/'test_metrics.csv');roll=pd.read_csv(out/'rollout_summary.csv')
    manifest=json.loads((out/'data_manifest.json').read_text());selected=json.loads((out/'locked_policies.json').read_text())
    rm=pd.read_csv(out/'reward_test_metrics.csv');cal=pd.read_csv(out/'reward_test_calibration.csv')
    base=metrics[metrics.arm=='baseline'].iloc[0]
    means=metrics.groupby('arm',sort=False).mean(numeric_only=True).reset_index()
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    fig,ax=plt.subplots(figsize=(8.5,5),layout='constrained')
    for arm,g in metrics.groupby('arm',sort=False):
        ax.scatter(g.predicted_completion10,g.ndcg10,color=COLORS[arm],s=42,alpha=.8,label=LABELS[arm])
        if arm!='baseline':
            ax.annotate('',xy=(g.predicted_completion10.mean(),g.ndcg10.mean()),xytext=(base.predicted_completion10,base.ndcg10),arrowprops={'arrowstyle':'->','color':COLORS[arm],'alpha':.5})
    ax.set(xlabel='Predicted completion · top 10',ylabel='Observed next-day NDCG@10',title='Reward–relevance trade-off · locked policies')
    ax.grid(alpha=.15);ax.legend(frameon=False,fontsize=8)
    fig.savefig(reports/'tradeoff.png',dpi=180);plt.close(fig)
    fig,axes=plt.subplots(1,3,figsize=(12,3.8),layout='constrained')
    for ax,metric,title in zip(axes,['repeat10','top1_share','catalog_coverage10'],['Seen-item share in top 10','Largest top-1 item share','Catalog coverage in top 10']):
        for i,(_,r) in enumerate(means.iterrows()):
            ax.barh(i,r[metric],color=COLORS[r.arm])
        ax.set_yticks(range(len(means)),[LABELS[a] for a in means.arm],fontsize=8);ax.invert_yaxis();ax.set_title(title,fontsize=10);ax.grid(axis='x',alpha=.15)
    fig.savefig(reports/'behavior.png',dpi=180);plt.close(fig)
    fig,axes=plt.subplots(1,3,figsize=(12,3.8),layout='constrained')
    for arm,g in roll[roll['mode']=='append_action'].groupby('arm',sort=False):
        v=g.groupby('step').mean(numeric_only=True)
        for ax,field in zip(axes,['completion','repeat_in_generated','disagreement']):ax.plot(v.index,v[field],marker='o',label=LABELS[arm],color=COLORS[arm])
    for ax,title in zip(axes,['Predicted completion','Repeated generated item','Reward ensemble disagreement']):
        ax.set(title=title,xlabel='Generated step',xticks=range(1,6));ax.grid(alpha=.15)
    axes[0].legend(frameon=False,fontsize=7)
    fig.savefig(reports/'rollout.png',dpi=180);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(9,3.7),layout='constrained')
    axes[0].plot([0,1],[0,1],ls='--',color='#aaa');axes[0].plot(cal.predicted,cal.observed,'o-',color='#147d6f')
    axes[0].set(xlabel='Predicted completion',ylabel='Observed completion',title='Reward calibration · logged test pairs')
    axes[1].bar(rm.model,rm.rmse,color=['#aaa','#293e55','#147d6f']);axes[1].set(ylabel='RMSE',title='Held-out reward error');axes[1].tick_params(axis='x',labelrotation=15)
    fig.savefig(reports/'reward_model.png',dpi=180);plt.close(fig)
    validation=pd.read_csv(out/'validation_policies.csv')
    fig,axes=plt.subplots(1,2,figsize=(10,3.7),layout='constrained')
    for arm,g in validation.groupby('arm',sort=False):
        axes[0].plot(g.step,g.predicted_completion10,'o-',color=COLORS[arm],label=LABELS[arm])
        axes[1].plot(g.step,g.ndcg10,'o-',color=COLORS[arm],label=LABELS[arm])
    axes[0].axhline(selected['reference']['predicted_completion10'],color=COLORS['baseline'],ls='--')
    axes[1].axhline(selected['reference']['ndcg10'],color=COLORS['baseline'],ls='--')
    axes[0].set(title='Validation predicted completion',xlabel='RL updates',xticks=[20,40,80])
    axes[1].set(title='Validation NDCG@10',xlabel='RL updates',xticks=[20,40,80])
    for ax in axes: ax.grid(alpha=.15);ax.legend(frameon=False,fontsize=7)
    fig.savefig(reports/'validation.png',dpi=180);plt.close(fig)
    training=pd.read_csv(out/'rl_training.csv')
    fig,axes=plt.subplots(1,2,figsize=(10,3.8),layout='constrained')
    for arm,g in training[training.seed==42].groupby('arm',sort=False):
        if arm!='replay_kl_control': axes[0].plot(g.step,g.rollout_objective.rolling(10,min_periods=1).mean(),label=LABELS[arm],color=COLORS[arm])
        axes[1].plot(g.step,g.kl.rolling(10,min_periods=1).mean(),label=LABELS[arm],color=COLORS[arm])
    axes[0].set(title='Arm-specific training return',xlabel='Update',ylabel='10-update moving mean')
    axes[1].set(title='KL to the supervised policy',xlabel='Update',ylabel='Nats per visited state')
    for ax in axes: ax.grid(alpha=.15);ax.legend(frameon=False,fontsize=7)
    fig.savefig(reports/'training.png',dpi=180);plt.close(fig)
    display=means[['arm','ndcg10','recall10','predicted_completion10','repeat10','top1_share']].copy();display['arm']=display.arm.map(LABELS)
    detailed=metrics[['arm','seed','ndcg10','recall10','filtered_ndcg10','predicted_completion10','predicted_continuation10','catalog_coverage10','repeat10','pair_overlap10','genre_diversity10']]
    c=json.loads((out/'continuation_test.json').read_text());ab=json.loads((out/'reward_history_ablation.json').read_text())
    best=means[means.arm!='baseline'].sort_values('predicted_completion10',ascending=False).iloc[0]
    ci=pd.read_csv(out/'paired_bootstrap.csv')
    observations=[]
    control=means[means.arm=='replay_kl_control'].iloc[0]
    for _,r in means[means.arm!='baseline'].iterrows():
        bounds=ci[(ci.arm==r.arm)&(ci.metric=='ndcg10')]
        positive=int((bounds.ci_low>0).sum()); negative=int((bounds.ci_high<0).sum())
        observations.append(f"- **{LABELS[r.arm]}:** изменение NDCG относительно базы {r.ndcg10-base.ndcg10:+.4f}, относительно Replay + KL {r.ndcg10-control.ndcg10:+.4f}; изменение предсказанной доли просмотра {r.predicted_completion10-base.predicted_completion10:+.4f}. Доля просмотренных объектов в top-10 — {r.repeat10:.1%}, доля самого частого top-1 — {r.top1_share:.1%}. Парный интервал NDCG целиком выше нуля у {positive}/3 seed, ниже нуля у {negative}/3.")
    interpretation='\n'.join(observations)
    step_rows=[]
    for arm,g in roll[roll['mode']=='append_action'].groupby('arm',sort=False):
        first=g[g.step==1].mean(numeric_only=True);last=g[g.step==5].mean(numeric_only=True)
        step_rows.append({'arm':LABELS[arm],'reward_step1':first.completion,'reward_step5':last.completion,'repeat_step5':last.repeat_in_generated,'uncertainty_step1':first.disagreement,'uncertainty_step5':last.disagreement})
    steps_table=table(pd.DataFrame(step_rows),list(step_rows[0]))
    examples=json.loads((out/'rollout_examples.json').read_text())
    titles={m['token']:m['title'] for m in json.loads((out/'catalog.json').read_text())['metadata']}
    sample_user=examples[0]['user_id']
    case=[]
    for example in examples:
        if example['user_id']==sample_user and example['seed']==42 and example['mode']=='append_action':
            case.append('- **'+LABELS[example['arm']]+':** '+' → '.join(titles.get(t,'Unknown') for t in example['generated']))
    case_text='\n'.join(case)
    text=f'''# Sequential recommendation with reward optimization

Исследование того, как дообучение с подкреплением меняет рекомендации GPT-2 на KION. Токен — идентификатор фильма или сериала; награду оценивает отдельная модель отклика по истории пользователя и предложенному объекту.

Главный вопрос — можно ли увеличить ожидаемую долю просмотра, сохранив релевантность и разнообразие. Поэтому вместе с reward измеряются попадание в следующие реальные просмотры, повторы, концентрация выдачи и поведение при генерации пяти объектов.

![Компромисс награды и релевантности](reports/tradeoff.png)

## Результаты

Каждый вариант RL повторён с seed 42, 17 и 2026. Таблица показывает среднее по этим трём запускам; базовая модель одна, с seed 42. Это разброс RL при фиксированном предобучении, а не три независимых запуска всей системы.

{table(display,list(display.columns))}

Для базовой модели NDCG@10 = **{base.ndcg10:.4f}**, предсказанная доля просмотра = **{base.predicted_completion10:.4f}**. Максимальное среднее предсказанное вознаграждение среди вариантов имеет **{LABELS[best.arm]}**: **{best.predicted_completion10:.4f}**, NDCG@10 = **{best.ndcg10:.4f}**. Это описание уже зафиксированных результатов, не выбор модели по тесту.

Предсказанная доля просмотра не является измеренной полезностью рекомендаций. Истинный отклик на новые выдачи в логе отсутствует. Ошибки модели награды, смещение наблюдавшейся политики и изменение контекста могут объяснять рост этой величины.

[Подробный разбор](reports/EXPERIMENT.md) · [Все метрики](results/test_metrics.csv) · [Парные доверительные интервалы](results/paired_bootstrap.csv) · [Протокол](PROTOCOL.md)

## Что изменилось

{interpretation}

Значимость в отдельных запусках не означает устойчивость между периодами. При проверке многих вариантов эти интервалы служат диагностикой, а не основанием выбрать победителя на тесте.

## Модель отклика

Ансамбль из трёх небольших нейросетей предсказывает два сигнала: долю просмотра и наличие следующего взаимодействия в течение семи дней. Входы — усреднённые обучаемые эмбеддинги истории, эмбеддинг кандидата, их произведение, длина контекста. Идентификатор пользователя, длительность и доля просмотра кандидата во входы не входят.

{table(rm,list(rm.columns))}

![Качество модели вознаграждения](reports/reward_model.png)

Reward model оценивается на реально наблюдавшихся тестовых парах с историей до даты каждой пары; это отдельный протокол от фиксированного контекста ранжирования.

Перемешивание историй между тестовыми примерами даёт RMSE **{ab['rmse']:.4f}**. Эта диагностика показывает, насколько модель использует соответствие истории и кандидата; сама по себе она не подтверждает качество контрфактических оценок.

Для предсказания продолжения: AUC **{c['auc']:.4f}**, Brier **{c['brier']:.4f}**, Brier константы из обучения **{c['train_mean_baseline_brier']:.4f}**, наблюдений с полным горизонтом **{c['n']}**. Это условная корреляция на просмотренных объектах, не оценка эффекта рекомендации на возвращаемость.

## Протокол

- **Данные:** {manifest['cohort_users']:,} пользователей с не менее чем пятью взаимодействиями до 12 июля; {manifest['cohort_rows']:,} пар во всей выбранной истории. Будущая активность не участвовала в выборе пользователей.
- **Каталог:** 1 000 наиболее частых объектов только по обучению; ещё два токена — padding и неизвестный объект. Покрытие обучающих взаимодействий: {manifest['train_target_catalog_coverage']:.1%}; покрытие тестовых целей: {base.target_catalog_coverage:.1%}.
- **Обучение:** цели до 12 июля 2021; {manifest['examples']['train']['examples']:,} примеров после заранее заданного ограничения.
- **Выбор базовой модели и reward model:** 12–25 июля. **Контексты RL:** история до 26 июля. **Выбор RL-checkpoint:** 2–8 августа. **Тест:** 9–22 августа.
- **Оценка:** фиксированная история на начало периода; релевантны все объекты первого следующего наблюдаемого дня. NDCG и Recall — по {int(base.active_queries):,} активным пользователям; поведение выдачи — по всем {int(base.queries):,}. Отдельно сохранён NDCG по всем пользователям.

У событий внутри дня нет известного порядка: целевой пример использует только предыдущие дни. Порядок исторических событий с одинаковой датой определяется ID объекта. Неизвестные цели остаются в знаменателях Recall и NDCG. Основная оценка не исключает просмотренные объекты, чтобы обнаруживать повторы; в CSV дополнительно есть результаты с фильтром просмотренного.

## Дообучение

Модель — `GPT2LMHeadModel` с нуля: 2 блока, 4 головы внимания, размер эмбеддинга 64, контекст 32. Базовое обучение использует next-item cross-entropy. Восьми эпохам соответствует восемь проверок на development; сохраняется лучший NDCG.

REINFORCE генерирует четыре независимые пятишаговые траектории на каждый из 16 контекстов. Полная дисконтированная награда использует γ = 0.95. Для снижения дисперсии применяется leave-one-out baseline по остальным трём траекториям того же пользователя. Награда и baseline отделены от градиентов; обучается только политика.

Сравниваются чистая доля просмотра с KL-коэффициентами 0.01 и 0.15; консервативная награда со штрафами за повторы и расхождение ансамбля; смесь доли просмотра и вероятности продолжения. Во всех вариантах одинаковый supervised replay. Контроль `Replay + KL` не использует reward-gradient.

Checkpoint выбирается на 20, 40 или 80 обновлении по максимальному validation NDCG среди вариантов, не снижающих предсказанную долю просмотра относительно базы. Если таких нет, выбирается максимальный NDCG и фиксируется невыполнение ограничения. Для повторных seed используется уже выбранное число шагов. [Конфигурации](results/locked_policies.json) зафиксированы до финальной оценки.

![Выбор продолжительности обучения на validation](reports/validation.png)

Этот график относится к выбору моделей до теста. Он показывает цену дополнительных шагов обучения; пунктир — исходная GPT-2.

![Обучение и отклонение от исходной модели](reports/training.png)

Кривые возврата используют разные функции награды, поэтому их абсолютные уровни между вариантами несопоставимы. Показан seed 42 и скользящее среднее по 10 обновлениям.

![Изменения поведения](reports/behavior.png)

## Пять шагов вперёд

![Поведение авторегрессионной выдачи](reports/rollout.png)

После каждого выбора токен рекомендации добавляется в историю. Это **воображаемое действие**, а не подтверждённый просмотр. Вторая диагностика оставляет исходный контекст неизменным. При жадном выборе без фильтров такой контроль повторяет один и тот же объект — ожидаемое свойство процедуры, а не ошибка измерения.

Пошаговые оценки награды, расхождение ансамбля, повторы внутри списка и совпадения с реальной историей сохранены в [таблице](results/rollout_summary.csv). Есть [примеры выдачи](results/rollout_examples.json). Эти измерения позволяют увидеть дрейф, но не заменяют модель переходов пользователя.

## Воспроизведение

```bash
git clone https://github.com/KarinaSH7/kion-sequential-rl.git
cd kion-sequential-rl
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-lock.txt
python -m pytest -q
```

Результаты и checkpoints исходного эксперимента уже включены. Перед полным повтором сохраните их:

```bash
mv results results_reference
PYTHON=.venv/bin/python bash run.sh
```

Нужен интернет для загрузки KION; дальше расчёт локальный, на CPU с четырьмя потоками. Сырые данные не хранятся в Git. Контрольные суммы источника проверяются при загрузке. `src.report` заново строит графики и отчёт из результатов без обучения.

| Код | Назначение |
|---|---|
| `src/data.py` | Когорта, словарь, временные примеры и цензурирование |
| `src/models.py` | GPT-2 и модель отклика |
| `src/train.py` | Предобучение, reward fitting, RL и выбор checkpoints |
| `src/evaluate.py` | Ранжирование и диагностика поведения |
| `src/final.py` | Оценка зафиксированных политик и bootstrap |
| `src/audit.py` | Временные проверки и независимый пересчёт метрик |

## Ограничения

KION содержит агрегированные пары и последнюю дату просмотра, а не журнал показов и сессий. Условная доля просмотра наблюдается только после взаимодействия. Нет вероятностей показов, отрицательных показов и причинной разметки удержания, поэтому IPS/DR-оценка новой политики здесь не заявляется.

В метаданных нет надёжной длительности фильма. Нельзя вычислять её из `total_dur / watched_pct`: это использует отклик и агрегированные повторные просмотры. Поэтому смещение к коротким объектам остаётся непроверенным риском, а не установленным результатом.

Эксперимент ограничен активной когортой, популярным каталогом, одной временной проверкой и небольшой GPT-2. Предсказания ансамбля могут ошибаться согласованно; его расхождение — эвристика неопределённости, а не гарантия защиты от эксплуатации reward model.

## Источники

- [KION dataset](https://github.com/irsafilo/KION_DATASET), фиксированная версия `{manifest['source_revision']}`. Petrov et al., *MTS Kion Implicit Contextualised Sequential Dataset for Movie Recommendation*, 2022.
- [Petrov & Macdonald, 2024](https://arxiv.org/abs/2403.04875): GPTRec и оптимизация критериев за пределами точности. Здесь используется самостоятельная компактная постановка, без teacher–student стадии из статьи.
- [Matveev et al., 2026](https://arxiv.org/abs/2607.02818): последовательная оптимизация и off-policy correction. Наш on-policy sampling внутри фиксированной модели награды не воспроизводит их off-policy estimator.
- [GPT-2 implementation](https://huggingface.co/docs/transformers/model_doc/gpt2).

Карина Шаля · T Lab · задание 2.
'''
    (ROOT/'README.md').write_text(text)
    detail=f'''# Разбор эксперимента

## Полные результаты

{table(detailed,list(detailed.columns))}

## Выбор checkpoint

{table(validation[['arm','step','ndcg10','predicted_completion10','repeat10']],['arm','step','ndcg10','predicted_completion10','repeat10'])}

## Пять последовательных действий

{steps_table}

## Пример выдачи

Первый пользователь из фиксированной случайной выборки, seed 42. Пример выбран по порядку в выборке, а не по интересности результата.

{case_text}

## Функции вознаграждения

Пусть m(s,a) — средняя прогнозируемая доля просмотра, u(s,a) — стандартное отклонение трёх моделей, c(s,a) — прогноз следующего взаимодействия за 7 дней, I — индикатор того, что объект уже встречался в полной известной истории или текущей сгенерированной последовательности. N(a) — нормированный −log частоты объекта по обучению, ограниченный единицей.

- Completion: r = m.
- Conservative: r = m − 0.5u − 0.3I + 0.1N.
- Continuation mix: r = 0.5m + 0.5c − 0.3I.
- Replay + KL: r = 0, только replay и KL.

R = Σ γ^t r_t / Σ γ^t. Advantage для траектории — её R минус среднее R остальных трёх независимых траекторий с тем же исходным контекстом. Loss = −mean(A × Σ log π(a_t|s_t)) + β mean(KL(π||π_ref)) + 0.2 CE_replay. KL вычисляется точно по словарю на посещённых сгенерированных состояниях; градиент через распределение этих дискретных состояний для регуляризатора не берётся. Это обычный per-state surrogate, а не точный градиент всего регуляризованного распределения траекторий.

Политика работает без dropout при sampling и расчёте log-probability, но с включёнными градиентами. Reference и reward networks заморожены. Нет top-k sampling, скрытой подстановки релевантных объектов или обновления reward model в ходе RL.

## Что проверяется офлайн

Наблюдаемые следующие взаимодействия позволяют сравнивать retrieval-релевантность на общем фиксированном срезе. На реально просмотренных объектах можно оценивать калибровку доли просмотра и прогноза продолжения. По выдачам можно измерять повторяемость, популярность, концентрацию, жанровое разнообразие и величину отклонения от базовой политики.

Однако оценка m(s,a) для нового a не имеет наблюдаемой истины. Даже хороший RMSE на логах не гарантирует перенос на выдачи оптимизированной политики. Низкое расхождение ансамбля тоже не исключает общую систематическую ошибку. Улучшение предсказанной награды следует читать как успешную оптимизацию заданной функции, а не рост удовлетворённости.

## Долгосрочные цели

Второй head прогнозирует наличие взаимодействия в любой из следующих семи дат после наблюдавшегося объекта. Если полный горизонт не помещается в период, пример исключается из loss этого head и его оценки. Повторные события той же даты не считаются возвращением.

Полезные дополнительные сигналы: число активных дней за 7/30 дней, вероятность следующей сессии, время до возврата с survival-цензурированием, завершённость пользовательской задачи, отрицательные реакции и явная удовлетворённость. Здесь агрегированный журнал не позволяет надёжно различить сессии, показы и потребление. Продолжение может отражать исходную вовлечённость пользователя, а не действие рекомендации.

Для проверки причинного эффекта нужны случайное назначение политики, журнал показов и фактических реакций, достаточный горизонт и анализ по пользователям. Первичные метрики — возвращаемость/активные дни и удовлетворённость; ограничения — скрытия, повторы, концентрация каталога и качество опыта. Период и размер эксперимента определяются заранее через мощность; краткосрочный watch fraction не должен единолично определять победителя.

## Многошаговое смещение

Авторегрессионная генерация с добавлением каждого предложенного объекта предполагает потребление с вероятностью один. В действительности пользователь может пропустить, закончить сессию, перейти к другому объекту или вернуться позже. Reward model обучена на реальных историях; пять искусственно добавленных объектов меняют распределение её входов.

Контроль с замороженным контекстом изолирует эффект добавления токенов при одинаковом жадном выборе. Он не моделирует реакции пользователя. Более реалистичное продолжение потребовало бы отдельной модели переходов: выбор/пропуск/завершение, обновление истории только после потребления, время между событиями. Без этих данных суммарная пятишаговая награда — оптимистичная model-based диагностика.

## Статистическая интерпретация

[Парный bootstrap](../results/paired_bootstrap.csv) использует 2 000 повторных выборок пользователей. Для релевантности берутся активные пользователи, для предсказанной награды — вся когорта запросов. В каждом сравнении совпадают пользователи и reference. Интервалы условны на обученных моделях; они не включают неопределённость источника данных, reward model, временных периодов и базового предобучения. Поправка на множественные сравнения не делается, поэтому по ним не выбирается «победитель».

Контроль Replay + KL нужен потому, что дополнительные supervised updates сами могут менять релевантность. Проверка гипотезы о чистом вкладе награды требует сопоставлять RL не только с исходной GPT-2, но и с этим контролем.

## Воспроизводимость и дальнейшая работа

Исходная версия датасета и SHA-256 фиксированы. Checkpoints, seed, конфигурации, метрики по пользователям, top-10 списки и примеры пятишаговой генерации сохранены. Аудит независимо пересчитывает Recall/NDCG и проверяет строгие даты историй и цензурирование продолжения.

Следующие эксперименты: несколько rolling-origin тестов; более широкий каталог; отдельный seed предобучения в каждом повторе; reward model с независимым аудитом на другой временной выборке; отдельное сравнение многошагового и одношагового обучения; реальные журналы показов с propensity для корректной off-policy оценки. Эти эксперименты не проводились в данной работе.
'''
    (reports/'EXPERIMENT.md').write_text(detail)
    print('Saved README, detailed report and six figures.')


if __name__=='__main__':main()
