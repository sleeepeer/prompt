# prompt

10.17-10.30的调研内容

*此为论文列表，详细内容在子目录*

**所有论文都已经上传了pdf文件，后面也标了链接*



## survey1 低层general，高层specific

* **AutoLR: Layer-wise Pruning and Auto-tuning of Learning Rates in Fine-tuning of Deep Networks**	[pdf](https://ojs.aaai.org/index.php/AAAI/article/view/16350)

  之前你给我那篇，里面提了浅/深层general/specific的问题，引出了以下两篇讨论这个的文章 10.18

* **How transferable are features in deep neural networks?**	[pdf](https://proceedings.neurips.cc/paper/2014/hash/375c71349b295fbe2dcdca9206f20a06-Abstract.html)

  量化了general和specific，讨论了转移过程（在哪些层） ，也有固定浅层feature的做法 10.18

* **Visualizing and Understanding Convolutional Networks**	[pdf](https://link.springer.com/chapter/10.1007/978-3-319-10590-1_53)

​		可视化CNN各层学习到了哪些feature，实现上提出DCNN反卷积网络，对解释general/specific有帮助 10.19

* **From Generic to Specific Deep Representations for Visual Recognition**  [pdf](https://www.cv-foundation.org/openaccess/content_cvpr_workshops_2015/W03/html/Azizpour_From_Generic_to_2015_CVPR_paper.html)



## survey2 类似cache feature的工作（transfer learning + prompt）

* **（survey1）How transferable are features in deep neural networks?**	[pdf](https://proceedings.neurips.cc/paper/2014/hash/375c71349b295fbe2dcdca9206f20a06-Abstract.html)

* **Convolutional Neural Networks for Medical Image Analysis: Full Training or Fine Tuning?** 	[pdf](https://ieeexplore.ieee.org/abstract/document/7426826)

	提出incremental fine-tuning，从最后层开始依次递增做fine-tune

* **迁移学习transfer learning博客** https://www.zhihu.com/question/41979241、

  列了几篇DNN迁移学习好文，domain adaptaion好文

  ---

  ### 待考察

  * https://arxiv.org/abs/2210.00990
  * https://arxiv.org/abs/2110.07904
  * https://dl.acm.org/doi/abs/10.1145/3534678.3539249?casa_token=tdaVtTr9PmMAAAAA:4xBmXPrcvwAx4iCxaov2ZFSSNZcKhkzbLHX-5PxytYI4KHf3O5qmDKQsnX6hcOBwdWun5_dCbaUiwSk



## other

* 
