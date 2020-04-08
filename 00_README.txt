

    Predictor - Actor (- Recollector) モデルと負の学習
    (Created: 2020-02-03, Time-stamp: <2020-02-06T16:40:21Z>)


** 概要

今回も失敗した実験である。改造した MountainCarContinuous のアルゴリズ
ミックな学習で、アルゴリズミックな中では、確かに部分的な成功もあるので
あるが、しかし、ランダムで単純な学習と比較した場合、早さはもちろん、学
習が進むためのステップ数でも、最終的な到達点でも、アルゴリズミックな方
法のほうが負けてしまったのだった。悔しいが、今回のアルゴリズミックな方
法は、まったくの無駄だったと結論せざるを得ない。何か活かせる方法を今後、
考えたい。

そのアルゴリズミックな学習とは、まず第一に、「環境」+「行動の提案」
→「予想」を行う Predictor を学習しておき、その誤差逆伝播法を利用して
「環境」+「予想」→「行動の提案」を行う Actor を学習するというもの。逆
伝播がちゃんと使えるかが問題としてあったが、概ね使えるという実験結果を
得た。

次に、Predictor と Actor の同時学習も試みた。その際に、Actor の「行動
の提案」と、Predictor から逆伝播された「行動の提案」がどちらが良いかと
いうのが以前私がレポートした「競争的学習」のようになっていると考えた。
それについて「負の学習」を試み、それが学習を早くするという結果を得た。

しかし、この二つのアルゴリズミックな学習の結果・部分的な成功は、先に述
べたように、ランダムで単純な学習と比べると負けていたのだった。


** Predictor - Actor - Recollector のアイデア

ある日、「カエルがある力でジャンプするとして、目的の距離を行くために、
想起した行動の行動結果がその目的よりも小さければ、より力を込めるように
する…といった学習をしなければならない。」…というのを例として考えてい
た。

どれぐらいの力だとどれぐらい進むかを知るために、まず、いろいろなシチュ
エーションで、(ときに同じシチュエーションで、)ランダムに学習することが
必要ではないか。

次に、特定のリアルな問題において、ジャンプしてみて、それが届かなかった
とする。このとき、どれぐらい力が足りなかったかを知れねばならない。それ
をランダムに学習したところから導き出せるか？

「どれぐらいの力だとどれぐらい進むか」学習するというのは、「環境」と
「行動の提案」が与えられたときに、それがどのようなあたらしい状態や環境
になるか「予想」をできるようになるということである。

しかし、行動する上で実際に欲しいのは、むしろ、「環境」と「予想」が与え
られたときに正しい「行動の提案」を行うことである。

「環境」+「行動の提案」→「予想」を行うのを、第一マシン、または、
Predictor と本稿では呼ぶことにする。

それに対し、「環境」+「予想」→「行動の提案」を行うのを、第二マシン、
または、Actor と本稿では呼ぶ。

第二マシン(Actor)が、提案したことを実行する…上の例 では、ジャンプして
みたとして、それがうまく行動できなかったとする。このとき第一マシン
(Predictor)の学習結果から、どれぐらい力が足りなかったかを知り、それを
第二マシンの学習に利用できるだろうか？

このとき第一マシンの誤差逆伝播法(の途中経過)から得られる差分(微分)の情
報が利用できないだろうかというのがアイデアである。こういう逆伝播の使い
方は、Deep Dream などでも使っていたはずである。

第二マシンの「行動の提案」から「実際の結果」が得られるという形になって
いる。と「環境」と「行動の提案」から第一マシンが行う「予想」の出力と第
二マシンに入力された「予想」は特に学習の進んでいない段階では必ずしも同
じではない。

第二マシンの学習に欲しいのは、「実際の結果」をもたらすはずだった「本当
の行動提案」である。入力された「欲しい予想」と出力された「あいまいな予
想」が違うとしても、欲しいのは「本当の行動提案」に近いものであるから、
それを得るための差分は「あいまいな予想」に「欲しい予想」- 「実際の行動
結果」を足したものを正例として逆伝播すれば得られるのではないかと考えた。
そして、それを第二マシンの正例として突っ込み学習する。


少し話を戻す。カエルのジャンプと同時に、おもちゃの自動車の自動運転につ
いてもどうすればいいか考えていた。「右に曲がる」や「S字に曲がる」とい
う操作を学習することを考える。

風景が流れていて、どういう風景の流れならば、「右に曲がる」であったり
「S字に曲がる」であったりするかということを学習できないか。風景の流れ…す
なわち「環境」の時間的入力に対し、それがどれだけ「右に曲がる」というこ
とを示しているかを 1 から 0 で学習するということを考える。

そしてそうして学習したマシンから、逆にある最初の「環境」が与えられたと
き、上の Deep Dream じみた逆伝播を使い、次の環境の「予想」を生成する。
そして「環境」と「予想」を入力とする第二マシンに「行動の提案」をさせて、
実際の行動をさせればいい。

…といったことを考えた。このマシンを、第三マシン、または、Recollector
と呼ぶ。この Predictor - Actor - Recollector で運動することを学習した
いというのが私の一時みた夢であった。強化学習と目指す分野は似ているが、
方法論が少し違う。

第三マシンについては、本稿ではこれ以上突っ込んだ議論をしないが、第三マ
シンでも、Deep Dream 的な逆伝播を使うことが想定されている。まずは、簡単
な例で、上の第二マシンの学習について、逆伝播を使った学習がうまくいくか
確めたいと考えた。それが、今回の実験の動機である。


運動の簡単な例として、OpenAI Gym の中の、カエルのジャンプにも、おもちゃ
の自動車にも少しひっかかる感じの MountainCarContinuous を少し改造しな
がら利用する。MountainCarContinuous は車で丘を昇り切るのを目的とした強
化学習用の例である。通常の MountainCar は 1 か -1 のデジタル値を「行動
の提案」として要するのに対し、MountainCarContinuous は 1.0 から -1.0
までの連続値を入力する。

私の第一マシン・第二マシン・第三マシンは、連続値の入力を前提として考え
てきたので、Continuous なものを選択した。

それを改造したというのは、元の MountainCarContinuous は丘に登り切るには、
100 ステップ以上の入力が必要であるが、そんなに長い入力だと学習が難しい。
そこで、車の power や max_speed をいじった。

また、元は、探索(exploration)と利用(exploitation)をしながら、いける状
態を探っていくのであるが、そんなまだるっこしいことはせず、自由に状態を
選んで、そこからランダムに行動した結果というのを、得られるように無理矢
理している。

さらに、第三マシン的なものからの、「環境」と「予想」から「行動」を得る
とき、元の線形のままだとあまり学習している感じがないので、入力は、正負
負符号を一旦外したものを二乗したものに正負符号をかけなおしたものにして
いる。


実装は、Python 3 + TensorFlow を用いた。TensorFlow はバージョンが 2.0
になり、かなりこれまでとは違ったプログラミングが必要とされる。そこで、
TensorFlow 1.15 でテストしたものと、TensorFlow 2.0 でテストしたものの二
種類の実装を行った。総じて、TensorFlow 2.0 を使ったものは、コードは美し
いが、とても遅くなった。私のプログラミングの腕の問題もあるだろうが…。


** 実験: MountainCarContinuous を最適化で無理やり解いてみる。

まず最初に MountainCarContinuous を使うにあたって、それがどういうもの
か、どれほどの難しさを持つものか、改造等はうまくいくか…を知るために、
それを scipy.optimize を使って解いてみることにした。

特に裏技的なものを使わず、action の列を入力として最適化する。
MountainCarContinuous を env.seed を固定し、env.reset したあと、
env.step し、それが doneになれば、そのときの reward を値としてそれを最
大化するように最適化する。短い action の列で done になれば良く、done に
なった以降は action が 0.0 であればなおよし、となるように reward を組ん
だ。

しかし、うまくいかない。車が上がり切らないときの reward に傾斜がうまく
付いてないのが問題なのだろう…と考え、done になる前は、車の速度の絶対
値を足し合わせたものを reward とし、done になったら、速度に関係なくし
てみた。

すると、action の列は長さ 150 くらいを指定して Dual Annealing 法で最適
化すると、いろいろ値は変わるが、ある回は、108 の長さで、done になるよう
答えが得られた。

<source>
$ python e01_car_optimizer.py --length=150
(…)
OptForStep: iterated 1000 times score=9279.999999997199
[ 1.00000000e+00  1.00000000e+00  1.00000000e+00  1.00000000e+00
(…)
 -2.06879154e-06  2.99421647e-07]
Done at  108
</source>

Windows ではアラームが鳴ったあと、次のアニメーションが得られる。(アラー
ムを鳴らすには、pip install pygame する必要がある。元のサイズは
600x400 だが 300x200 にしてある。)

{{e01_car_optimizer.gif}}

しかし、action の長さが 100 を越えるというのは長過ぎで、これを学習して
いくのは難し過ぎるだろう。…ということで、あまりやってはいけないことか
もしれないが、MoutainCarContinuous の env.unwrapped の power と
max_speed をいじって、もっと短い解を出すことを考える。

すると、--power=0.5 --max-speed=1.0 で、action の列の長さが 5 ぐらいで
いろいろな動きを出せることがわかった。いろいろな動きを出すために、
reward の計算に速度を加味することにする。

できた、optimizer に渡す関数は次のようになっている。

<source>
class CarOptimizer:
(…)
    def _opt_main (self, l):
        self.env.seed(self.seed)
        observation = self.env.reset()
        sum_velocity = 0
        for i, a in enumerate(l):
            observation, reward, done, info = self.env.step([a])
            position = observation[0]
            velocity = observation[1]
            sum_velocity += np.abs(velocity)
            if done:
                done = i + 1
                break
        if done:
            for i in range(done, len(l)):
                reward -= (l[i] ** 2) * 0.1
            if self.velocity_reward == "sum":
                reward = reward * 100 + sum_velocity * 10000
            elif self.velocity_reward == "last":
                reward = reward * 100 + np.abs(velocity) * 10000
            elif self.velocity_reward == "neg":
                reward = reward * 100 - np.abs(velocity) * 100
            else:
                reward -= 10 * (done - 1) / len(l)
                reward *= 100
        else:
            reward = min(sum_velocity, 100)
        return - reward
</source>

オプションで --length=5 --power=0.5 --max-speed=1.0
--velociti-reward=last などとすると、最後に最高速度となるよう一旦戻っ
てから坂を降りるアニメが表示される。これを今回の標準的な設定とする。


** 実験: Predictor をまず学習し、次に逆伝播を利用して Actor を学習する。

通常、MountainCarContinuous は、env.step に action を順次与えて次の状
態を得ることはできるが、任意の状態に対し、任意の action を与えて次の状
態を得るということはできない。が、env.unwrapped.state に値を代入し、次
に env.step することで、「任意の状態に対し、任意の action を与えて次の
状態を得る」ということは試したところ可能だった。

ダーティだがそれを使い、ランダムに発生させたいろいろな状態と action に
ついて、「環境」+「行動の提案」→「予想」を行う Predictor を学習する。
tf.keras を使って実装しているが、これは特に説明を要するものではない。
先に説明したとおり、デフォルトで、--power=0.5 --max-speed=1.0 になって
いる。

<source>
$ python e02_train_predictor.py
(…)
Epoch 300/300
300/300 [==============================] - 1s 2ms/step - loss: 0.0117
300/300 [==============================] - 0s 2ms/step - loss: 0.0116
Test Score:  0.011614195782070359
(…)
</source>

さて、次に「環境」+「予想」→「行動の提案」を行う Actor の実装である。
逆伝播した勾配を「行動の提案」に足したものを「本来すべきだった行動」と
するのであるが、勾配の値はとても小さくしかも一回足しただけでは別に記録
した「本来すべきだった行動」とかけはなれすぎていることがわかった。

そこで、勾配を「行動の提案」に足したものをもう一度、Predictor で予想し
て、それに関しても逆伝播したものを新たに足すべき勾配とする…というのを
数回(今回は50回)することにした。

逆伝播するときの正例は、何度も繰り返しても問題ないようにということで、
「予想」を正例とすることにした。これにより、「実際の行動」を必要とせず、
Predictor の記憶だけを頼りに、Actor を学習することになった。

肝心の部分のソースは TensorFlow 2.0 では次のようになる。

<source>
class Actor:
(…)
    @tf.function
    def _calc_temp_action(self, current, purpose, action):
        pout = self.predictor.model \
            (tf.concat([current, action], axis=1))
        grad = tf.gradients(K.mean(K.square(purpose - pout)),
                            [action])
        return action \
            - self.grad_coeff \
            * tf.cast(tf.shape(action)[0], tf.float32) * grad[0]
    
    def calc_pseudo_action(self, current, purpose, action):
        purpose = purpose.astype(np.float32)
        current = current.astype(np.float32)
        action = action.astype(np.float32)
        for i in range(self.pseudo_action_iteration):
            action = self._calc_temp_action(current, purpose, action).numpy()
        return action
</source>

current が「環境」、purpose が「予想」、pseudo_action が「本来すべきだっ
た行動」になる。逆伝播の一ステップが _calc_temp_action で、
calc_pseudo_action でそれを数回している。tk.gradients で勾配を求め、基
本はそれを元の action から引くのであるが、勾配 grad の値は総じて小さ過
ぎるので、tf.shape(action)[0] すなわち batch_size をかけ、さらに
grad_coeff として 0.1 をかけている。

ここで得られた pseudo_action を正例の値として、Actor の出す action に
ついて学習する。

学習はあまり良くはないが、それなりに進む。次のような結果のあと、学習に
関するグラフが表示される。

<source>
$ python e03_train_actor_tf1.py
(…)
Epoch:  0  Step Loss:  0.03041661264685293  tLoss:  0.2086267611458999  pLoss:  0.14513424122091764
Epoch:  1  Step Loss:  0.005070856858510524  tLoss:  0.16392257231882168  pLoss:  0.14909930848102884
(…)
Epoch:  18  Step Loss:  0.0004511116956806897  tLoss:  0.12698073674694926  pLoss:  0.1281853588127608
Epoch:  19  Step Loss:  0.0004345568039085871  tLoss:  0.13047537194886394  pLoss:  0.13151520528443897
(…)
</source>

pseudo_action と action の差(の二乗和)が、Step Loss の部分で、本来すべ
き行動だった true_acton と action の差が tLoss、true_action と
pseudo_action の差が pLoss と表示されている。見ると Step Loss が小さい
値なのに対し、tLoss と pLoss は近い値となっている。つまり、
pseudo_action による true_action への近似はさほどうまくいっていないの
がわかる。にもかかわらず、学習自体は進んでいるようだ。

これが、うまく学習していることを確かめるため、Recollector もどきを実装
する。

e01_car_optimizer.py のように scipy.optimize で、どういう actionの列な
らば、良い reward が得られるか見つける。その最初の action が満たすべき
「環境」+「予想」を入力として Actor が出した「行動の提案」にしたがって
行動する。すると、それはなすべきだった「予想」とは少し違った値になる。
それを考慮して、現在、達成された部分を固定入力として、残りの部分につい
て最適化する。そして、次の action を決定する…というのを長さ 5 か done
になるまで繰り返す。

それの結果が次のようになる。

<source>
$ python e04_test_recollector_tf1.py
(…)
OptForStep: iterated 1000 times score=19999.99324834986
[-0.53283685 -0.8899985   0.98263411  0.98913868  0.02598394]
current:  [-0.43112599  0.        ]  purpose:  [-0.6982291  -0.26710311]  act:  -0.858027  act^2:  -0.7362103007310452
(…)
OptForStep: iterated 1000 times score=19999.999900136318
[0.00316012]
current:  [0.3022419 1.       ]  purpose:  [0.6 1. ]  act:  -0.0064819604  act^2:  -4.2015810832518596e-05
Done:  [-0.858027, -0.9698391, 1.0400538, 1.039655, -0.0064819604]

</source>

次のようなアニメーションを出力する。

{{e04_test_recollector_tf1_1.gif}}

先に述べたように実際の入力は action を二乗したものである。その一方、
optimizer への入力は、二乗をしてないものである。OptForStep のあとの行
の数値の列が次からとるべき action の列を表している。current が「環境」、
purpose が「予想」で、そこから Actor が導き出したのが act であるが、そ
れが合うべきは act を二乗した act^2 が、上の行の最初の数値と合致してい
ればうまくいっている証拠である。

最後に達したあとは、スピードが制限されるので、最後があまり合ってないの
はあまり気にするところではない。だが、最初の部分も、それほど精度良く合っ
てないのは少し問題があるかもしれない。

とはいえ、そこそこはうまくいっているようだ。

しかし、これは --power=0.5 --max-speed=1.0 が地味に効いてこうなってい
ることがわかっている。本来の --power=0.0015 --max-speed=0.07 の場合、
Actor の学習はなぜか完全に失敗する。


** Predictor - Actor の負の学習のアイデア

Actor は「実際の行動」を必要とせずに学習ができるのであるが、そもそも
Predictor を学習するとき、「環境」+「行動の提案」→「予想」が必要なの
だから、その「環境」「行動の提案」「予想」をそのまま使って「環境」
+「予想」→「行動の提案」の Actor の学習もできるはずであることに、上の
実験が終ったあと、やっと気付いた。

それを知って、ちょっとヘコんだのだが、思い直し、だったら、Actor と
Predictor を同時に学習することにして、そのとき、逆伝播も利用すれば学習
効率が良くなることはないかと考えた。もちろん、学習の実行スピードは遅く
なるだろうが、ステップ数で見た場合は少ないステップ数でできるようになる
かもしれない…。そもそも、元々のアイデアは Recollector を回しながら、
Actor と Predictor が少しずつ賢くなっていくというものだった。

「実際の行動」actual は貴重なデータで、それを学習して有効利用するのは
もちろんだが、Predictor の逆伝播データも有効に利用できるならしたい。

「行動提案」action を実行して得られる actual と 逆伝播から得られた「本
来とるべきと考えられる行動」pseudo_action を実行して得られる pactual
があると考える。actual と pactual のどちらか一方のみが得られるとする。
まずは pactual を得る方向を考える。

Predictor の「予想」は predicted とし、Actor の入力として与えられる
「予想」purpose とは微妙に異なるとする。

Predictor の pseudo_action を実行して pactual を得たあと、predicted と
違いがあったとき、Predictor は言わば予想を失敗していると言える。
pseudo_action と元の action の間に対立的(競争的)関係が生じていると考え
れるのではないか。

私は以前、競争的学習に関してブログにレポートを書き、そこで「負の学習」
をすることで学習が早くなることを報告した。今回も「負の学習」の枠組が持
ち込めるのではないか。

Predictor が負けたとき、元の action のほうが purpose に近かったとして、
0.5 * (pacutual + purpose) を正しいものとして paction について
Predictor は学習すればよいのだろうか。ちなみに Predictor が勝ったとき
は、すなわち pactual と purpose が近いということだから、pactual に関す
る学習だけで十分であろう…。

さらに、Predictor は、pseudo_action を計算するまで、gradients の計算で
いくつもの action predicted のペアを出力している。その i 番目の出力に
ついて、 ((1+i)/N) * (0.5 * (purpose + pactual) - predicted) +
predicted[i] を線形に分配する形で学習してみてはどうだろう？

Predictor が負けたとき Actor はまず「負の学習」の要素、負けた
pseudo_action と action の距離がとても遠いときは無視し、ある程度近いと
きは逆側に倒すという、- (pseudo_action - action) * exp(-
(pseudo_action - action) ** 2) の要素にさらに、負けた度合を表す
tanh((pactual - purpose) ** 2) をかけたものを学習すれば良いのではない
か。

…と考えた。


** 実験: Predictor - Actor の負の学習。

まず、負けたときの Predictor については、基本的に pactual が正しいはず
だから、0.5 * (pacutual + purpose) ではなく pactual を正例として学習す
ることにした。

それ以外の細かい部分は以下のソースを見てもらおう。

<source>
def comp_train (cenv, pred, actor, current, purpose):
    action = actor.model(np.concatenate([current, purpose], axis=1)).numpy()
    action = np.clip(action, -1.0, 1.0)

    pactions = []
    predicteds = []
    purpose = purpose.astype(np.float32)
    current = current.astype(np.float32)
    action = action.astype(np.float32)
    pseudo_action = action
    pactions.append(action)
    for i in range(actor.pseudo_action_iteration):
        pseudo_action, predicted = actor.calc_temp_action \
            (current, purpose, pseudo_action)
        pseudo_action = pseudo_action.numpy()
        predicted = predicted.numpy()
        pactions.append(pseudo_action)
        predicteds.append(predicted)
    pactions.pop()

    pseudo_action = np.clip(pseudo_action, -1.0, 1.0)
    pactual = np.array(list([
        cenv.calc_next_state(state[0], state[1], act[0])
        for state, act in zip(current, pseudo_action)
    ]))

    pred.model.optimizer.lr = ARGS.predictor_lr
    pred_loss = pred.model.train_on_batch(
        np.concatenate([current, pseudo_action], axis=1),
        pactual
    )
    actor.model.optimizer.lr = ARGS.actor_lr
    actor_loss = actor.model.train_on_batch(
        np.concatenate([current, pactual], axis=1),
        pseudo_action
    )

    pinputs = []
    poutputs = []
    delta = 0.5 * (pactual + purpose) - predicted
    for i, (pa, pr) in enumerate(zip(pactions, predicteds)):
        y = delta * ((i + 1) / len(pactions)) + pr
        x = np.concatenate([current, pa], axis=1)
        pinputs.append(x)
        poutputs.append(y)
    pinputs = np.concatenate(pinputs, axis=0)
    poutputs = np.concatenate(poutputs, axis=0)
    pred.model.optimizer.lr = ARGS.predictor_comp_lr
    pred_comp_loss = pred.model.train_on_batch(pinputs, poutputs)

    negact = action - (pseudo_action - action)\
        * np.exp(- ((pseudo_action - action) / 2.0) ** 2) \
        * np.tanh(np.mean(((pactual - purpose) /
                           np.array([[cenv.env.power * 2,
                                      cenv.env.max_speed * 2]])) ** 2,
                          axis=1, keepdims=True))
    actor.model.optimizer.lr = ARGS.actor_comp_lr
    actor_comp_loss = actor.model.train_on_batch(
        np.concatenate([current, purpose], axis=1),
        negact
    )

    return action, pseudo_action, pactual, predicted, \
        pred_loss, actor_loss, \
        pred_comp_loss, actor_comp_loss
</source>

実行すると、

<source>
$ python e05_train_comp_pa_tf1.py
(…)
Epoch: 0  Step taLoss: 0.3878614084396775 paLoss: 0.35301687094676204 apLoss: 0.032118778706205935 ppLoss: 0.046547495388974375 aLoss: 0.0831309718824923 pLoss: 0.07953674641437829 acLoss: 0.0019627396604300885 pcLoss: 0.03394529583863914
(…)
Epoch: 49  Step taLoss: 0.06110515049354414 paLoss: 0.06390214986747228 apLoss: 0.0008193471449747029 ppLoss: 0.0005126261923907525 aLoss: 0.006991444195737131 pLoss: 0.0012225553154712543 acLoss: 0.00021323761208350334 pcLoss: 0.0017119957141403575
(…)
</source>

ちなみに、これまでは 1 epoch あたりの --steps=300 だったのを
--steps=100 にして学習スピードの変化をわかりやすくしている。

これについても車を動かしてみる。

<source>
$ python e04_test_recollector_tf1.py --reset-sleep=30 --end-sleep=10
(…)
OptForStep: iterated 1000 times score=19996.03566781198
[ 4.43710465e-01  9.32296151e-01  6.29629410e-01 -1.08925159e-04
  1.13950094e-04]
current:  [-0.48084667  0.        ]  purpose:  [-0.2593112   0.22153547]  act:  0.7458111  act^2:  0.5562342040049231
OptForStep: iterated 1000 times score=19995.035565063645
[ 0.75392373  0.70076171 -0.02632415 -0.0684365 ]
current:  [-0.2593112   0.22153547]  purpose:  [0.44965954 0.65270886]  act:  1.097463  act^2:  1.204425062141027
OptForStep: iterated 1000 times score=17742.963602522585
[-4.30153397e-06  1.50327865e-06 -3.45129766e-06]
current:  [0.44965954 0.65270886]  purpose:  [0.6        0.77611176]  act:  0.629207  act^2:  0.39590146777244684
Done:  [0.7458111, 1.097463, 0.629207]
(…)
</source>

アニメーションは次のようになる。

{{e04_test_recollector_tf1_2.gif}}

ちゃんと動いてはいる。

「負の学習」を使うこと／使わないことの変化を見てみる。
e05_train_comp_pa_tf*.py に --actor-comp-lr=0 や --predictor-comp-lr=0
を足したりして --save-history で history を出力したあと、
e07_show_histories.py を使って比べてみた。

{{e07_show_histories.py.png}}

「本来の行動」true_action と「行動の提案」action の差の変化のグラフで
ある。w ac が Actor に関する「負の学習」アリ、no ac がナシ。w pc が
Predictor に関する「負の学習」的なものがアリ、no pc がナシ。…になって
いる。Predictor に関する「負の学習」はまったく効果がなさそうだが、
Actor に関する「負の学習」は学習を早くする効果はあるようだ。

Actor の Optimizer には Adam を使っていて、Predictor の Optimizer には
RMSProp を使っている。それらを使うことで、--actor-comp-lr=0 のように学
習率を 0 に設定しても、悪影響は残る形になっている可能性がある。そこで、
そもそも「負の学習」は使わないが、pseudo_action は求めて、それで
pactual を求め学習するプログラムも作った。それが
e06_train_simple_pa_tf*.py である。

その結果が上のグラフでは no ac no pc 2 として記録されている。それによ
ると、no ac no pc の 2 でないほうと大差はないようだった。

が、気になって、さらに、pseudo_action は求めないものを作って
e06_train_simple_pa_tf*.py で --simplest として指定できるようにしてみ
た。すると、これはほとんど学習が進まない。さらに気が付いて、action 自
身をランダムにして、その acutual なものを学習する --random-action とい
うのを作ってみた。--simplest --random-action とした結果が、上のグラフ
の random の線である。ちなみにこれが、先に述べた「環境」「行動の提案」
「予想」をそのまま使って Predictor と Actor を同時に学習するのに相当す
る。

なんと「負の学習」を使ったものより、random なもののほうが、学習のスピー
ドが早く、最終的な結果も良くなっている。

確かにアルゴリズミックにやる分には多少、「負の学習」の効果があると言え
るのかもしれないが、しかし、ランダムなものにはまったくかなわない。私が
やったことは基本的に無駄だったことがわかった。


** 実験: 逆伝播をするとき Optimizer を使ってみる。

逆伝播をするときに求めた勾配を引くとやっていたが、これは SGD (確率的勾
配降下法)に相当する方法だった。その代わりに、他の Optimizer を使うこと
にすれば、もっとよい pseudo_action が得られ、それにより学習がよりよく
進むのではないかと考えた。

やってみたのが e08_train_actor_tf*.py である。ちなみに、TensorFlow 2.0
版はここではいっそう遅くなってしまう。もっとよいプログラミングのしかた
があるのだろうか？

Adam を使った結果が次になる。(e02_train_predictor_tf1.py を実行しなお
したあと。)

<source>
$ python e08_train_actor_tf1.py
(…)
Epoch:  0  Step Loss:  0.04491752951095502  tLoss:  0.303568049987199  pLoss:  0.22991417511206289
Epoch:  1  Step Loss:  0.017004587134967247  tLoss:  0.23702449656821842  pLoss:  0.2101400177893807
(…)
Epoch:  18  Step Loss:  0.004106538979879891  tLoss:  0.07723008512396802  pLoss:  0.07821667089819768
Epoch:  19  Step Loss:  0.003960698268686732  tLoss:  0.07978376407564954  pLoss:  0.08039417448980536
(…)
</source>

確かに、上の e03_train_actor_tf1.py の結果と比べ、Step Loss が多めに出
ていることから、pseudo_action が true_action に近付いていると考えられ、
最終的な結果も良いが、e05_train_comp_pa_tf1.py などと比べて最終的な結
果までが良くなっているとは言えない。

効果は限定的なわりにとても時間がかかるので、e05_train_comp_pa_tf1.py
に Optimizer を取り込んだ実験は行わなかった。


** 実験: 初期アイデアに近い形で逆伝播を使った Actor の学習をする。

e03_train_actor_tf1.py では、「実際の行動」をしないで良いのが言ってみ
れば、ウリなのであるが、初期のアイデアでは「実際の行動」をして、
purpose を Predictor の正例とする代わりに purpose - pactual +
predicted を正例とすることを考えていたのだった。それを試してみる。

<source>
class Actor:
(…)
    def calc_pseudo_action(self, actual, current, purpose, action):
        purpose = purpose.astype(np.float32)
        current = current.astype(np.float32)
        action = action.astype(np.float32)
        actual = actual.astype(np.float32)
        
        predicted = self.predictor.model \
            (np.concatenate([current, action], axis=1)).numpy()
        purpose = purpose - actual + predicted
        for i in range(self.pseudo_action_iteration):
            action = self._calc_temp_action(current, purpose, action).numpy()
        return action
</source>

_calc_temp_action はそのままに、calc_pseudo_action を上のように変えた。

結果、

<source>
$ python e09_train_actor_tf1.py
(…)
Epoch:  0  Step Loss:  0.032415725216269495  tLoss:  0.22955234666241817  pLoss:  0.1640542957588051
Epoch:  1  Step Loss:  0.008300049300305545  tLoss:  0.20067297562621292  pLoss:  0.16981464672676458
(…)
Epoch:  19  Step Loss:  0.00041168005739261083  tLoss:  0.159099302919301  pLoss:  0.15813861866697904
(…)
</source>

e03_train_actor_tf1.py より少し悪い。本当の purpose ではなく偽の
purpose を使っているのに結果がまずまずなのが意外だ。

さらに同時に predictor も途中から train していく形にする。

<source>
$ python e09_train_actor_tf1.py --train-predictor
(…)
Epoch:  0  Step Loss:  0.0356204578311493  tLoss:  0.23872311356703033  pLoss:  0.1685584666394354
Epoch:  1  Step Loss:  0.006451640491529058  tLoss:  0.207461743984704  pLoss:  0.18587929238919432
(…)
Epoch:  19  Step Loss:  0.0009023256706132088  tLoss:  0.10098316846040539  pLoss:  0.10093547380286093
(…)
</source>

今回は多少良くなっているが、むしろ悪くなることもあった。「実際の行動」
をすることの利点があまりない。最初に実験したときはさらにうまくいってい
るようにみえたこともあったので、謎である。

current purpose pseudo_action の組を学習しているわけだが、当然、
current actual action の組も学習できる。それが --train-actual で試せる
のだが、それを試すのは例が増えて良いことのはずなのに、なぜかうまくいか
なかった。

さて、ランダムに学習するとどうなるかも比較対象としてやっておこう。

<source>
$ python e09_train_actor_tf1.py --train-true
(…)
Epoch:  0  Step Loss:  0.20823655802756547  tLoss:  0.20823655905723823  pLoss:  0.20823655905723823
Epoch:  1  Step Loss:  0.10306025767078002  tLoss:  0.10306025858104645  pLoss:  0.10306025858104645
(…)
Epoch:  19  Step Loss:  0.04027129391441122  tLoss:  0.040271293929725914  pLoss:  0.040271293929725914
(…)
</source>

かなり良いのがわかる。e03_train_actor_tf1.py もはるかに凌駕する。

やはりランダムで単純な学習には勝てないようだ。


** 実験: まず行動する Predictor - Actor 同時学習。

e09_train_actor_tf1.py が最初、うまくいっているように見えたので、同じ
ような感じで e05_train_comp_pa_tf*.py を改造してみることにした。アイデ
アのところで pactual をとるか、actual をとるかどちらかにすべきだと書い
て、上では pactual をとった。それを actual をとるようにしてみた。

<source>
def comp_train (cenv, pred, actor, current, purpose):
    action = actor.model(np.concatenate([current, purpose], axis=1)).numpy()
    action = np.clip(action, -1.0, 1.0)
    actual = np.array(list([
        cenv.calc_next_state(state[0], state[1], act[0])
        for state, act in zip(current, action)
    ]))

    pactions = []
    predicteds = []
    purpose = purpose.astype(np.float32)
    current = current.astype(np.float32)
    action = action.astype(np.float32)
    apredicted = pred.model(np.concatenate([current, action], axis=1)).numpy()
    ppurpose = purpose - actual + apredicted
    ppurpose = ppurpose.astype(np.float32)
    pseudo_action = action
    pactions.append(action)
    for i in range(actor.pseudo_action_iteration):
        pseudo_action, ppredicted = actor.calc_temp_action \
            (current, ppurpose, pseudo_action)
        pseudo_action = pseudo_action.numpy()
        ppredicted = ppredicted.numpy()
        pactions.append(pseudo_action)
        predicteds.append(ppredicted)
    pactions.pop()
    pseudo_action = np.clip(pseudo_action, -1.0, 1.0)
    ppredicted = pred.model(np.concatenate([current,
                                            pseudo_action], axis=1)).numpy()

    pred.model.optimizer.lr = ARGS.predictor_lr
    pred_loss = pred.model.train_on_batch(
        np.concatenate([current, action], axis=1),
        actual
    )
    actor_loss2 = 0
    if ARGS.actor_actual_lr != 0.0:
        actor.model.optimizer.lr = ARGS.actor_actual_lr
        actor_loss = actor.model.train_on_batch(
            np.concatenate([current, actual], axis=1),
            action
        )
    if ARGS.actor_pseudo_lr != 0.0:
        actor.model.optimizer.lr = ARGS.actor_pseudo_lr
        actor_loss2 = actor.model.train_on_batch(
            np.concatenate([current, purpose], axis=1),
            pseudo_action
        )
    if ARGS.actor_actual_lr == 0.0:
        actor_loss = actor_loss2

    pinputs = []
    poutputs = []
    delta = ppurpose - ppredicted
    for i, (pa, pr) in enumerate(zip(pactions, predicteds)):
        y = delta * ((i + 1) / (1 + len(pactions))) + pr
        x = np.concatenate([current, pa], axis=1)
        pinputs.append(x)
        poutputs.append(y)
    pinputs = np.concatenate(pinputs, axis=0)
    poutputs = np.concatenate(poutputs, axis=0)
    pred.model.optimizer.lr = ARGS.predictor_comp_lr
    pred_comp_loss = pred.model.train_on_batch(pinputs, poutputs)

    delta = np.mean(((purpose - ppredicted)
                     / np.array([[cenv.env.power * 2,
                                  cenv.env.max_speed * 2]])) ** 2,
                    axis=1, keepdims=True)
    negact = action - (pseudo_action - action)\
        * np.exp(- ((pseudo_action - action) / 2.0) ** 2) \
        * np.tanh(delta)
    actor.model.optimizer.lr = ARGS.actor_comp_lr
    actor_comp_loss = actor.model.train_on_batch(
        np.concatenate([current, purpose], axis=1),
        negact
    )

    return action, pseudo_action, actual, ppredicted, \
        pred_loss, actor_loss, \
        pred_comp_loss, actor_comp_loss
</source>

<source>
$ python e10_train_comp_pa_tf1.py
(…)
Epoch: 0  Step taLoss: 0.3814685874694013 paLoss: 0.3510731952808972 apLoss: 0.03335011591477158 ppLoss: 0.1042007241873584 aLoss: 0.006803616329525539 pLoss: 0.09291472819633782 acLoss: 0.0007054742167565564 pcLoss: 0.04612612306140363
Epoch: 1  Step taLoss: 0.2135844642353302 paLoss: 0.20242004115450826 apLoss: 0.009013494424028773 ppLoss: 0.016259977124444533 aLoss: 0.005820183390751481 pLoss: 0.01196747493930161 acLoss: 0.00011279224642748886 pcLoss: 0.02325278322212398
(…)
Epoch: 30  Step taLoss: 0.17215656649776917 paLoss: 0.1705528932492742 apLoss: 0.009837894612437283 ppLoss: 0.005611145104219588 aLoss: 0.00347990282374667 pLoss: 0.004043028907617554 acLoss: 0.00030812633010100397 pcLoss: 0.0056136824726127086
Epoch: 31  Step taLoss: 0.08931298785288176 paLoss: 0.08365105199825042 apLoss: 0.005519177598887847 ppLoss: 0.004018510048812091 aLoss: 0.0038912209670525044 pLoss: 0.0045223186875227835 acLoss: 0.0006336112073495315 pcLoss: 0.00427393484278582
(…)
Epoch: 49  Step taLoss: 0.0817590828406703 paLoss: 0.08292756372761005 apLoss: 0.0057550643172081485 ppLoss: 0.0026833485138182678 aLoss: 0.002109217004326638 pLoss: 0.0036345214530592784 acLoss: 0.00013027903481088287 pcLoss: 0.0040082720894133676
(…)
</source>

ちゃんと学習自体はできているようだが、e05_train_comp_pa_tf1.py に比べ、
学習は遅い。

実行してから考えるよりも、考えてから実行するほうが効果があるといったと
ころか。

「負の学習」を行わないのも試してみる。

<source>
$ python e10_train_comp_pa_tf1.py --actor-comp-lr=0 --predictor-comp-lr=0
(…)
Epoch: 0  Step taLoss: 0.36242321337018857 paLoss: 0.2626778246352604 apLoss: 0.03957104109817539 ppLoss: 0.1297674158313778 aLoss: 0.0389227571268566 pLoss: 0.11617005173116922 acLoss: 0.0017395445259899133 pcLoss: 0.054832416027784346
Epoch: 1  Step taLoss: 0.2087694407952734 paLoss: 0.19658901270169354 apLoss: 0.01340919349405097 ppLoss: 0.021329397692980222 aLoss: 0.026018686406314374 pLoss: 0.018512654304504394 acLoss: 0.0020963922965165694 pcLoss: 0.03360302812652662
(…)
Epoch: 49  Step taLoss: 0.13558385605153886 paLoss: 0.13600217480471188 apLoss: 0.006495833143161361 ppLoss: 0.005773937960702605 aLoss: 0.002054563459387282 pLoss: 0.0032620200351811944 acLoss: 0.00012205636059206882 pcLoss: 0.005549291088827886
(…)
</source>

…ということで、学習は進まず、「負の学習」に効果があることがわかる。

しかし、いずれにせよ、上で述べたようにランダムで単純な学習にはかなうべ
くもない。


** 結論と今後の課題

アルゴリズミックな学習・「負の学習」を応用した学習を提案した。が、それ
をランダムで単純な学習と比較した場合、早さはもちろん、学習が進むための
ステップ数でも、最終的な到達点でも、アルゴリズミックな方法のほうが負け
てしまったのだった。

「行動の提案」action を「実際の行動」actual に移す同じだけの機会がある
なら、ランダムに action を選んで、その結果を学習するほど、効率的なこと
はない。…というのが結論になりそうだ。

逆にランダムにするには、同じだけの機会がない…というのはどういう場合か？
強化学習のように「探索」が関係してくれば違ってくるのだろうか？ わから
ない。そのあたりが今後の課題になるかもしれない。


逆伝播による「本来あるべき行動」の近似はあまりうまくいっていない。が、
学習はある程度進むようだ。なぜそうなるかの究明は今後の課題である。

「負の学習」は以前のアイデアは、勝ったか負けたかの二値の問題だったが、
今回は、どれだけ負けたかという度合を掛けている点が、新しい。ここは今回、
ほぼ唯一、今後有望なところかもしれない。

MountainCarContinuous の環境で、元々の --power=0.0015 --max-speed=0.07
にするとうまくいかなかった。ニューラルネットの入力や出力で normalize
をすれば良いのかもしれないが、確かめていない。細かいが、それも今後の課
題とは言える。

e09_train_actor_tf1.py で、Predictor を追加で学習しようとしたり、
current actual action の例を足して学習しようとすると、逆に学習結果が悪
くなるということがあった。偏った例が学習されるからかもしれないが、詳し
いことはわからない。その究明も今後の課題である。


** 感想

カエルのジャンプの井戸を超えるイメージと、おもちゃの自動車の両者を満た
すものとして、MountainCarConitnuous が与えられていたのは天啓のようにも
感じた。が、それを結局はうまく活かせなかったのが残念だ。

「負の学習」を応用できる…といったあたりでかなりテンションが上がったの
だが、落ち着いてランダムで単純な学習と比較すると全く負けていることがわ
かりショックを受けた。実験は失敗だったと評価できる。

この先はまったく見えない。この先を考えるのは、少し時間がかかるかもしれ
ない。他のことをまずしたい。


** 参考

Python と TensorFlow でコードを書くにあたっては様々なサイトを参考にし
たが、とにかくわからないところを急ぎで探っていることが多く、細かく覚え
ていないため、申し訳ないが割愛する。

  * 《機械学習の練習のため競争的な学習の実験をしてみた》。

    http://jrf.cocolog-nifty.com/software/2019/01/post.html


  * 《ニューラルネットで負の学習: 競争的な学習の実験 その２》。

    http://jrf.cocolog-nifty.com/software/2019/05/post-3c97df.html

  * [cocolog:91382428]。＞田中＆富谷＆橋本「ディープラーニングと物理
    学」に目を通した。その感想とは関係ないが、ずっと気にしているおもちゃ
    の自動車の自動運転がらみのアイデアをブレイン・ストーミング的にここ
    で少し考える。＜ Predictor - Actor - Recollector モデルのアイデア
    は最初、ここに書いた。
    
    http://jrf.cocolog-nifty.com/statuses/2019/10/post-b58749.html

  * [cocolog:91609276]。＞曽我部東馬「強化学習アルゴリズム入門」と伊
    藤多一 他「現場で使える！ Python 深層強化学習入門」を読んだ。ソー
    スがあるのがありがたい。二冊を交互に読むことでアルファ碁ゼロが私の
    強化学習のイメージに近いことがわかった。一方だけではわからなかった。
    ＜ 強化学習は、Predictor - Actor - Recollector モデルとはまた違う
    ものだが、今後、参考にするかもしれない。今回の Keras のパラメータ
    などはこれらの本のサンプルプログラムを参考にしている。

    http://jrf.cocolog-nifty.com/statuses/2020/01/post-ed7a36.html


** 著者

JRF ( http://jrf.cocolog-nifty.com/software/ )


** ライセンス

私が作った部分に関してはパブリックドメイン。 (数式のような小さなプログ
ラムなので。)

自由に改変・公開してください。


(This document is written in Japanese/UTF-8.)
