## （survey1）transfer （NIPS 2014 talk）

> homepage：[Jason Yosinski](https://yosinski.com/transfer)
>
> paper：How transferable are features in deep neural networks?
>
> author：Jason Yosinski,1 Jeff Clune,2 Yoshua Bengio,3 and Hod Lipson4
>
> code：[yosinski/convnet_transfer: Code for paper "How transferable are features in deep neural networks?" (github.com)](https://github.com/yosinski/convnet_transfer)
>
> ---



## Full Training or Fine Tuning?

> paper：Convolutional Neural Networks for Medical Image Analysis: Full Training or Fine Tuning?
>
> author：Nima Tajbakhsh*, Member, IEEE*, Jae Y. Shin, Suryakanth R. Gurudu, R. Todd Hurst, Christopher B. Kendall, Michael B.Gotway, and Jianming Liang*
>
> ---
>
> main：
>
> * 对几个不同的医学cv任务，从只微调最高层开始递增（incremental fine-tuning），发现**不同任务需要微调的程度不同**，**全量微调不一定比只微调部分层效果好**。
> * 微调程度还与可用于训练的数据量有关（training data）；training data大的时候 *全量微调 > full train > 浅微调* ，随着data量变小，全量微调比full train会越来越好
> * 全量微调的收敛速度（convergence speed）比full train一个CNN更快
>
> point：
>
> * 不同task需要fine-tune的程度不同，有些只需要FT深层或者中层，有些则需要全局微调，所以需要做“Layer-wise fine-tuning”
> * 也有提到general/specific
> * source和target task差距大时只调深层是不行的（domain adaptation）
> * **提出技巧：从只调最高层开始一层层加（incremental fine-tuning）**
> 
>![](survey2.assets/屏幕截图 2022-10-24 153055.png)

