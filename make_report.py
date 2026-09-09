# -*- coding: utf-8 -*-
"""探索の全記録を単一 HTML にまとめる"""
import sys
sys.path.insert(0, '.')
from REPORT_DATA import EXPERIMENTS, SUBMISSIONS

CAT_COLOR = {"baseline":"#6b7280","診断":"#8b5cf6","model":"#0ea5e9","feature":"#f59e0b",
             "data":"#10b981","target":"#ec4899","後処理":"#84cc16","ensemble":"#ef4444",
             "CV":"#3b82f6","監査":"#a855f7"}
VERDICT_STYLE = {"keep":"v-keep","drop":"v-drop","診断":"v-diag","改善":"v-keep",
                 "(後に撤回)":"v-drop","未提出":"v-hold"}

rows = "\n".join(
    f'<tr><td class="mono">{i}</td><td class="mono dim">{d}</td><td>{n}</td>'
    f'<td><span class="chip" style="background:{CAT_COLOR.get(c,"#6b7280")}">{c}</span></td>'
    f'<td class="mono">{r}</td>'
    f'<td><span class="{VERDICT_STYLE.get(v,"v-hold")}">{v}</span></td><td class="dim">{note}</td></tr>'
    for i, d, n, c, r, v, note in EXPERIMENTS)

subs = "\n".join(
    f'<tr class="{"best" if "best" in nt else ""}"><td class="mono">{s}</td><td class="mono dim">{d}</td>'
    f'<td>{n}</td><td class="mono big">{sc}</td><td class="mono">{rk}</td><td>{nt}</td></tr>'
    for s, d, n, sc, rk, nt in SUBMISSIONS)

HTML = """<!doctype html><html lang="ja"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Store Sales Forecasting — 探索記録</title>
<style>
:root{--bg:#0f1115;--panel:#171a21;--line:#262b36;--fg:#e8eaed;--dim:#9aa3b2;--acc:#5eead4;--warn:#fbbf24;--bad:#f87171}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--fg);font-family:-apple-system,BlinkMacSystemFont,"Hiragino Sans","Noto Sans JP",sans-serif;line-height:1.7}
.wrap{max-width:1180px;margin:0 auto;padding:48px 24px 80px}
h1{font-size:30px;margin:0 0 6px;letter-spacing:-.02em}
h2{font-size:20px;margin:52px 0 14px;padding-bottom:8px;border-bottom:1px solid var(--line)}
h3{font-size:15px;margin:26px 0 8px;color:var(--acc)}
p,li{font-size:14px}
.sub{color:var(--dim);font-size:14px;margin:0 0 28px}
.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(190px,1fr));gap:12px;margin:22px 0}
.card{background:var(--panel);border:1px solid var(--line);border-radius:10px;padding:16px}
.card .k{color:var(--dim);font-size:11px;letter-spacing:.06em;text-transform:uppercase}
.card .v{font-size:26px;font-weight:600;margin-top:4px;font-variant-numeric:tabular-nums}
.card .n{color:var(--dim);font-size:12px;margin-top:2px}
table{width:100%;border-collapse:collapse;margin:14px 0;font-size:13px}
th{text-align:left;color:var(--dim);font-weight:500;font-size:11px;letter-spacing:.06em;
   text-transform:uppercase;padding:8px 10px;border-bottom:1px solid var(--line)}
td{padding:8px 10px;border-bottom:1px solid #1d222b;vertical-align:top}
tr:hover td{background:#1a1f28}
.mono{font-family:ui-monospace,SFMono-Regular,Menlo,monospace;font-variant-numeric:tabular-nums}
.dim{color:var(--dim)}
.big{font-size:15px;font-weight:600}
tr.best td{background:#12261f}
tr.best .big{color:var(--acc)}
.chip{display:inline-block;padding:2px 8px;border-radius:999px;font-size:11px;color:#0b0d11;font-weight:600}
.v-keep{color:#34d399;font-weight:600}.v-drop{color:var(--bad)}.v-diag{color:#c4b5fd}.v-hold{color:var(--warn)}
.pipe{display:flex;flex-wrap:wrap;gap:10px;align-items:stretch;margin:18px 0}
.leg{flex:1;min-width:150px;background:var(--panel);border:1px solid var(--line);border-radius:10px;padding:14px}
.leg .w{font-size:22px;font-weight:600;color:var(--acc);font-variant-numeric:tabular-nums}
.leg .t{font-size:13px;font-weight:600;margin:2px 0 4px}
.leg .d{font-size:11px;color:var(--dim);line-height:1.5}
.note{background:#12161d;border-left:3px solid var(--acc);padding:12px 16px;margin:14px 0;font-size:13.5px;border-radius:0 8px 8px 0}
.note.bad{border-left-color:var(--bad)}
.note.warn{border-left-color:var(--warn)}
code{background:#1d222b;padding:1px 6px;border-radius:4px;font-size:12.5px;
     font-family:ui-monospace,SFMono-Regular,Menlo,monospace}
pre{background:#12161d;border:1px solid var(--line);border-radius:8px;padding:14px;overflow-x:auto;font-size:12.5px}
.two{display:grid;grid-template-columns:1fr 1fr;gap:20px}
@media(max-width:820px){.two{grid-template-columns:1fr}}
footer{margin-top:60px;padding-top:20px;border-top:1px solid var(--line);color:var(--dim);font-size:12px}
</style></head><body><div class="wrap">

<h1>Store Sales — Time Series Forecasting</h1>
<p class="sub">Kaggle / エクアドル Corporación Favorita の店舗×商品ファミリ別 日次売上予測（RMSLE）<br>
探索記録 — 48実験 / 6提出</p>

<div class="grid">
  <div class="card"><div class="k">Final Public LB</div><div class="v">0.41533</div><div class="n">開始 0.42567 → −0.0103</div></div>
  <div class="card"><div class="k">Rank</div><div class="v">114<span style="font-size:15px;color:var(--dim)"> / 647</span></div><div class="n">上位 17.6%（開始時 142位）</div></div>
  <div class="card"><div class="k">実験数</div><div class="v">48</div><div class="n">採用 4 / 棄却 30 / 診断 14</div></div>
  <div class="card"><div class="k">最終転移率</div><div class="v">2.0</div><div class="n">CV −0.0018 → LB −0.0036</div></div>
</div>

<h2>1. タスクと制約</h2>
<table>
<tr><th>項目</th><th>内容</th></tr>
<tr><td>予測対象</td><td>54店舗 × 33商品ファミリ = <b>1,782系列</b> の日次売上</td></tr>
<tr><td>期間</td><td>train 2013-01-01〜2017-08-15（1,684日 / 300万行）→ test <b>2017-08-16〜08-31（16日）</b></td></tr>
<tr><td>評価</td><td>RMSLE = log1p 空間の RMSE → <b>log1p を目的変数にした L2 回帰</b>が指標と一致</td></tr>
<tr><td>与えられる未来情報</td><td>onpromotion / 祝日 / 原油価格は test 期間分あり。<b>transactions は 2017-08-15 まで</b>（当日値は使えない）</td></tr>
</table>

<h3>データから読み取れた構造</h3>
<ul>
<li><b>売上ゼロが全行の 31.3%</b>。BOOKS 96.9% / BABY CARE 94.1% と family で極端に違う</li>
<li><b>途中開店の店舗が8つ</b>（store 52 は 2017-04-20 開店で履歴4ヶ月弱）</li>
<li><b>1/1 はほぼ全店休業</b>（全店合計が通常の 1/100 以下）、12/25 は行ごと欠測</li>
<li><b>SCHOOL AND OFFICE SUPPLIES が8月に約6倍</b>へ立ち上がる（新学期需要）。test 期間はそのピーク中</li>
<li>周期性: 週末が山（日 463 / 木 284）、月内は 1日・30日・31日が山（給与日）</li>
</ul>

<h2>2. 最終パイプライン</h2>
<p>提出 #5（LB 0.41533）の構成。<b>5本すべてが同じ56特徴・同じ LightGBM</b> で、<b>学習単位だけが違う</b>。
特徴もハイパラも初期値から一切変えていない。</p>

<div class="pipe">
  <div class="leg"><div class="w">0.40</div><div class="t">base</div>
    <div class="d">全1,782系列を1モデルで学習</div></div>
  <div class="leg"><div class="w">0.20</div><div class="t">family別 (33)</div>
    <div class="d">商品カテゴリごとに独立学習<br>1分割 5.1万行</div></div>
  <div class="leg"><div class="w">0.20</div><div class="t">famgrp別 (4群)</div>
    <div class="d">売上規模3分位 × ゼロ率で集約<br>1分割 28万行</div></div>
  <div class="leg"><div class="w">0.20</div><div class="t">gseas別 (5群)</div>
    <div class="d">年次季節性3分位 × ゼロ率で集約<br>1分割 34万行</div></div>
</div>

<h3>共通設定</h3>
<pre>特徴 56個  : 静的6（store/family/city/state/type/cluster）
             カレンダー11（曜日/月内位置/<b>給与日</b>/年内日）
             祝日6（<b>locale を店舗の city/state と突き合わせ</b>）
             原油2 / onpromotion 4
             ラグ15（<b>lag16〜364</b>）+ 移動集約12（rmean7〜365, 同曜日平均, 前年同期比）
目的変数   : log1p(sales)      L2 損失（RMSLE と一致）
学習       : 2015-01-01 以降 / iter 900 / leaves 128 / lr 0.05 / <b>seed 3本平均</b>
予測       : expm1 して 0 クリップ
ホライズン : <b>日付基準 lag ≥ 16 を厳守</b>（test で入手不能な情報を使わない）</pre>

<h2>3. 検証設計 — ここが最大の争点だった</h2>
<div class="note bad"><b>失敗:</b> 最初の CV（2 fold）は <b>fold 間変動 0.069 に対して標準誤差 0.049</b>。
判定したい効果 0.002 の25倍で、<b>偶然で符号が決まる</b>状態だった。実際に2度、誤った施策を採用して LB を悪化させた。</div>

<div class="two">
<div>
<h3>cv3（最終形）</h3>
<ul>
<li>2016-06-01 から16日ステップで <b>27 fold</b>（重複なし）</li>
<li>各 fold: train = valid 開始日より前（提出時と同じ構造）</li>
<li><b>判定は必ず「同一 fold での対比較」</b> — fold 間変動を相殺</li>
<li>27fold 平均 <b>0.4406</b> ≒ 実測 LB <b>0.42567</b>（絶対水準が一致）</li>
</ul>
</div>
<div>
<h3>採否基準（3度の改訂を経た最終形）</h3>
<ol>
<li><b>勝率 ≥ 85%</b></li>
<li><b>中央値 ≤ 平均差</b>（利得が一様に分布し、少数 fold が平均を作っていない）</li>
<li><b>最悪 fold の悪化 ≤ +0.002</b></li>
</ol>
<p class="dim" style="font-size:12.5px">3つすべてを満たすこと。1つでも欠けると LB で逆転した実績がある。</p>
</div>
</div>

<h3>基準が改訂されていった経緯</h3>
<table>
<tr><th>施策</th><th>平均差</th><th>中央値</th><th>勝率</th><th>t</th><th>最悪 fold</th><th>実 LB</th></tr>
<tr><td>直近重み hl=365</td><td class="mono">−0.0044</td><td class="mono">−0.0026</td><td class="mono">78%</td><td class="mono">−1.76</td><td class="mono dim">—</td><td class="mono" style="color:var(--bad)">+0.0215 逆転</td></tr>
<tr><td>seed アンサンブル</td><td class="mono">−0.0020</td><td class="mono">−0.0016</td><td class="mono">85%</td><td class="mono">−3.68</td><td class="mono">−0.0008</td><td class="mono" style="color:#34d399">−0.0010 転移</td></tr>
<tr><td>family別ブレンド</td><td class="mono">−0.0048</td><td class="mono">−0.0039</td><td class="mono">96%</td><td class="mono">−1.32</td><td class="mono">−0.0008</td><td class="mono" style="color:#34d399">−0.0058 転移</td></tr>
<tr><td>4本ブレンド</td><td class="mono">−0.0018</td><td class="mono">−0.0016</td><td class="mono">92%</td><td class="mono">−6.17</td><td class="mono">+0.0010</td><td class="mono" style="color:#34d399">−0.0036 転移</td></tr>
<tr><td>再帰予測 0.2</td><td class="mono">−0.0042</td><td class="mono">−0.0026</td><td class="mono">93%</td><td class="mono">−4.50</td><td class="mono" style="color:var(--bad)">+0.0074</td><td class="mono" style="color:var(--bad)">+0.0063 逆転</td></tr>
</table>
<p class="dim" style="font-size:13px">同じ t≈−1.3〜−1.8 でも <b>hl=365 は中央値&lt;平均（少数 fold が利得を作る）</b>、
<b>family別は中央値&gt;平均（一様に改善）</b>で結果が正反対。最後の再帰予測は勝率93%・t=−4.50 を満たしたが、
<b>最悪 fold だけが7倍</b>で LB を悪化させた。</p>

<h2>4. 効いた原理 — 「学習単位だけを変える」</h2>
<p>特徴・ハイパラ・目的変数・後処理はすべて飽和していた。唯一機能したのは
<b>同じ特徴・同じアルゴリズムのまま、学習データの切り方を変えて脚を作る</b>ことだった。</p>

<table>
<tr><th>脚</th><th>1分割の行数</th><th>単体（base比）</th><th>ブレンド時</th><th>判定</th></tr>
<tr><td>store別 (54)</td><td class="mono">3.1万</td><td class="mono">+0.0399</td><td class="mono">勝率 0%</td><td class="v-drop">細かすぎ</td></tr>
<tr><td>cluster別 (17)</td><td class="mono">10万</td><td class="mono">+0.0178</td><td class="mono">勝率 65%</td><td class="v-drop">店舗側は無効</td></tr>
<tr><td><b>family別 (33)</b></td><td class="mono">5.1万</td><td class="mono">+0.0148</td><td class="mono">勝率 100%</td><td class="v-keep">採用</td></tr>
<tr><td>type別 (5)</td><td class="mono">34万</td><td class="mono">+0.0055</td><td class="mono">勝率 88%</td><td class="v-drop">gseas に劣後</td></tr>
<tr><td><b>famgrp別 (4群)</b></td><td class="mono">28万</td><td class="mono">−0.0016</td><td class="mono">勝率 100%</td><td class="v-keep">採用</td></tr>
<tr><td><b>gseas別 (5群)</b></td><td class="mono">34万</td><td class="mono dim">—</td><td class="mono">勝率 92%</td><td class="v-keep">採用</td></tr>
</table>

<div class="note"><b>3つの発見</b><br>
① <b>単体性能で脚を評価してはいけない</b> — family別は単体で base に負ける（勝率26%）が、混ぜると 27fold 中26勝。<br>
② <b>商品側の切り口だけが有効</b> — store / cluster / type はすべて失敗。「同じ商品は店舗をまたいで似た挙動をするが、同じ店舗の異なる商品は似ていない」という構造。<br>
③ <b>何を基準に集約するかで直交性が決まる</b> — 売上規模ベースの集約（g4/g6/g12）は互いに冗長だが、<b>季節性ベース（gseas）だけが独立した情報</b>を持つ。</div>

<h3>直交性と単体性能のトレードオフ</h3>
<table>
<tr><th>脚の作り方</th><th>単体（base比）</th><th>残差相関</th><th>結果</th></tr>
<tr><td>別アルゴリズム（Ridge / Croston / 前年同期）</td><td class="mono">+0.19 〜 +0.53</td><td class="mono">0.46 〜 0.76</td><td class="v-drop">単体が弱すぎて足せない</td></tr>
<tr><td>別ライブラリ（XGBoost）</td><td class="mono">−0.0060</td><td class="mono">0.977</td><td class="v-drop">直交しないので混ぜても効かない</td></tr>
<tr><td><b>学習単位を変える（family別など）</b></td><td class="mono">+0.0148</td><td class="mono dim">中程度</td><td class="v-keep">両立する「谷間」</td></tr>
</table>

<h2>5. 提出履歴</h2>
<table>
<tr><th>#</th><th>日付</th><th>構成</th><th>Public LB</th><th>順位</th><th>備考</th></tr>
""" + subs + """
</table>
<div class="note warn"><b>提出 #2 の教訓:</b> 3変更（学習期間 / 直近重み / seed数）を同時に入れて提出し、
<b>どれが原因か切り分け不能</b>になった。以後は必ず1変更ずつ提出し、#3 で seed 単独を検証して
「直近重みが犯人」と特定できた。</div>

<h2>6. 全実験記録</h2>
<table>
<tr><th>ID</th><th>日付</th><th>施策</th><th>分類</th><th>CV 結果</th><th>判定</th><th>要点</th></tr>
""" + rows + """
</table>

<h2>7. 棄却された軸（すべて cv3 で判定）</h2>
<div class="two">
<div>
<h3>特徴量</h3>
<ul>
<li><b>追加</b>: transactions ラグ / 集約系列 / 未来promo / 状態量 / カレンダー拡張 の38特徴 → 全グループ害</li>
<li><b>削減</b>: 下位10〜40個をカット → t=−0.85</li>
<li><b>表現変換</b>: 移動平均の比・フーリエ項 → t=−0.65、gain 合計 0.5%未満</li>
<li><b>ターゲットエンコーディング</b>: 3形態すべて不発（gain 比 2%）</li>
<li><b>祝日の種類別ダミー</b>: 各祝日が年3〜5サンプルしかなく学習不能（gain 0.00004）</li>
</ul>
<p class="dim" style="font-size:13px"><b>根拠:</b> gain の 76.9% を占める <code>rmean7</code> を削っても +0.0001。
上位5個（gain 92%）を削っても +0.0019。<b>特徴間の冗長性が極めて高く、木は少数の特徴で情報を汲み尽くしている</b>。</p>
</div>
<div>
<h3>モデル・目的・データ</h3>
<ul>
<li><b>ハイパラ</b>: 10設定すべて初期値以下（容量軸は飽和）</li>
<li><b>残差学習</b>（y − rmean7）: 重要度は分散したが精度は低下</li>
<li><b>two-part model</b>: 分類 AUC 0.988 でも oracle(−0.0377) は取れず</li>
<li><b>後処理</b>（ゼロ強制・確率縮小）: GBDT のヘッジが既に RMSLE 最適</li>
<li><b>階層モデル</b>: 独立予測の和が既に店舗合計と 1.5% 精度で一致</li>
<li><b>2ステップ</b>（transactions 予測）: 予測精度は十分だが情報が重複</li>
<li><b>学習期間カット</b>: 短くするほど単調に悪化</li>
<li><b>休業日の除外・補間</b>: +0.3460 で壊滅（休業は予測可能なパターン）</li>
<li><b>再帰予測</b>: CV で最強に見えたが LB で逆転（最悪 fold +0.0074）</li>
</ul>
</div>
</div>

<h2>8. 方法論として残ったもの</h2>
<ol>
<li><b>CV は「意思決定の道具」として精度要件を満たす必要がある。</b>
検出したい効果 0.002 に対し標準誤差が 0.049 では、何を測っても偶然。fold を増やすより
<b>同一 fold での対比較</b>（fold 間変動を相殺）のほうが効いた。</li>
<li><b>平均だけを見ない。</b> 中央値・勝率・最悪 fold を併記する。
同じ平均差でも「一様に改善」と「少数 fold が利得を作る」では転移可否が正反対。</li>
<li><b>1回の提出で1変更。</b> 束ねると切り分け不能になり、次の意思決定ができなくなる。</li>
<li><b>棄却時に「再挑戦条件」を書く。</b> 土台（モデル構造・予測方式）が変わると同じ施策の符号が変わる。
実際、ゼロ強制は direct では害だったが再帰予測では改善方向だった。</li>
<li><b>公開情報は検証してから採り入れる。</b> LB 0.37984 のノートの手法のうち、
祝日ダミー・ゼロ日補間・再帰予測は<b>いずれも我々の構成では悪化</b>した。
特に補間は +0.3460 で、検証せず真似していたら大事故だった。</li>
</ol>

<h2>9. 未着手 / 次の一手</h2>
<ul>
<li><b>base 枠を lgb と xgb で分割</b>（勝率85%・最悪+0.0016 で基準を満たすが未提出）</li>
<li>同じ発想を family別・famgrp・gseas の各脚にも適用</li>
<li>lags 範囲を変えた脚（短期のみ / 長期のみ）</li>
<li>NN 系（seq2seq / N-BEATS）— 単体性能と直交性を両立できる可能性が残る唯一の方向</li>
</ul>

<footer>
生成: 2026-09-09 ／ 全実験コードは <code>src/</code>、判断の記録は <code>experiments/</code>（EXP_LOG / KEEPERS / GRAVEYARD / IDEAS）に対応。
</footer>
</div></body></html>"""

open("REPORT.html", "w", encoding="utf-8").write(HTML)
print(f"wrote REPORT.html  ({len(HTML):,} bytes)")
