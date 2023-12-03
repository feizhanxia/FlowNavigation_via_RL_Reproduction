# A note for green hand in Q - learning
> Reproduction letter: Flow Navigation by Smart Microswimmers via Reinforcement Learning

## Introduction

这份[文档](A_note_for_green_hand_in_Q-learning.pdf)总结了部份入门强化学习的基本概念,但它并不是一份教材.教材往往内容丰富且知识体系完整,而这可能也常常对应着更加冗长的内容、有意义但并不实用的算法和超出需要的数学严谨.对于具备一定数学基础,却并不曾真正接触过强化学习,又渴望避开繁琐的细节,快速理解其核心数学定义并上手体验强化学习在科学中应用的同学,我希望这份笔记可以帮到你.

正如题目所写,这是一份通往理解Q-learning算法的学习笔记,虽然这里只介绍了一个算法,但学习这份笔记所带来的理解将会对理解其他强化学习算法十分有帮助.笔记第一部份会向你介绍强化学习中你会用到的的一些基本定义和定理,理解时序差分学习(一类强化学习算法)中很常用的一种算法_Q-learning.第二部份会向你展示一篇论文的复现作为应用的例子,查看它的代码并在这份代码基础上玩一玩会对你理解算法和消除陌生感有很大帮助.

注意,这份笔记建立的时候,我就和现在的你一样是新人,因而笔记中一定存在着大大小小的问题,如果你在阅读这份笔记过程中感到困惑或难以理解,请相信你自己,可以以此为线索对不理解的地方单独查找和学习,也可以先行阅读其他笔记或教材而将此笔记当作一份cheatsheet;也欢迎任何人指出问题或帮助改进和完善这一笔记,以帮助之后的人.

最后,祝你学得愉快,尽管可能会遇到困难,请永远不要放弃.

This [document](A_note_for_green_hand_in_Q-learning.pdf) summarizes some basic concepts of introductory reinforcement learning, but it is not a textbook. Textbooks are often rich in content with a complete knowledge system, which also often corresponds to longer content, meaningful but impractical algorithms, and mathematical rigor beyond necessity. For those with a certain mathematical foundation, who have not really encountered reinforcement learning, and wish to avoid cumbersome details to quickly grasp its core mathematical definitions and experience its application in science, I hope this note can help you.

As the title suggests, this is a learning note leading to an understanding of the Q-learning algorithm. Although only one algorithm is introduced here, the understanding gained from studying this note will be very helpful in understanding other reinforcement learning algorithms. The first part of the note will introduce you to some basic definitions and theorems used in reinforcement learning, and understand Q-learning, a commonly used algorithm in temporal difference learning (a type of reinforcement learning algorithm). The second part will show you the reproduction of a review letter as an example of application, examining its code and playing around with it based on this code will greatly help you understand the algorithm and eliminate unfamiliarity.

Please note, when I compiled these notes, I was a novice just like you are now. There are definitely issues, big and small, in the notes. If you feel confused or have difficulty understanding while reading these notes, trust yourself. You can use this as a clue to look up and learn about things you don't understand independently, or you can read other notes or textbooks first and treat these notes as a cheatsheet. I also welcome anyone to point out issues or help improve and perfect these notes, to assist those who come after.

Finally, I wish you a pleasant learning experience. Despite the potential difficulties, never give up.



**Below are the textbooks I referred to during my study (highly recommended, almost all of the content of the notes comes from them) and the paper reproduced in the second part, thanks to the authors of these works. I list them here for easy reference and learning.**

- "[Mathematical Foundations of Reinforcement Learning.](https://github.com/mathfoundationrl/book-mathmatical-foundation-of-reinforcement-learning)" by Shiyu Zhao
- "Flow Navigation by Smart Microswimmers via Reinforcement Learning", C. Simona and 
      G. Kristian and C. Antonio and B. Luca, [PhysRevLett.118.158004 (2017)](https://doi.org/10.1103/PhysRevLett.118.158004).

## Files

1. [`A_note_for_green_hand_in_Q-learning.pages`](A_note_for_green_hand_in_Q-learning.pages) and [`A_note_for_green_hand_in_Q-learning.pdf`](A_note_for_green_hand_in_Q-learning.pdf) are the Note in different kinds of format.
2. The code for reproduction of the letter is [`CodeForQ-learning.ipynb`](CodeForQ-learning.ipynb)
3. If you want to train several agents at the same time by multiprocessing with your multicore cpu, you may refer to [`multiprocessTrain.py`](multiprocessTrain.py)
4. You can find the plots I used to draw (drafts) in [`figs`](figs)
