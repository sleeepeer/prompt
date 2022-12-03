## P-tuning V2代码阅读笔记

> ### run.py
>
> run.py 流程概述
>
> 1. get_args()获取所有参数,主要是task/dataset_name和prefix等
> 2. 按照task_name进到对应的get_trainer()
> 3. 根据args获取trainer的过程：
> 	1. 根据模型名称或路径加载tokenizer
> 	2. 到tasks.superglue.dataset下获取dataset
> 		* dataset中含相关参数，比如只有copa是multiple_choice，其他都是sequence_classification
> 		* dataset中有数据预处理过程preprocess_function
> 	3. 根据模型名称或路径、dataset加载模型配置config
> 	4. 根据模型参数和模型配置加载模型(get_model)
> 		* 这里会根据TaskType和prefix等进行设置
> 	5. 根据以上设置的参数用training.trainer_base封装trainer返回
> 4. run.py最后有train、eval、predict模式的选择



那几篇CSDN读完，prompt和pt2网络结构



---

资料汇总：

[(103条消息) 知识图谱：【知识图谱问答KBQA（六）】——P-tuning V2训练代码解析_J_Xiong0117的博客-CSDN博客](https://blog.csdn.net/u013010473/article/details/122998825?ops_request_misc=%7B%22request%5Fid%22%3A%22166840262416782428630479%22%2C%22scm%22%3A%2220140713.130102334..%22%7D&request_id=166840262416782428630479&biz_id=0&utm_medium=distribute.pc_search_result.none-task-blog-2~all~sobaiduend~default-4-122998825-null-null.142^v63^wechat,201^v3^add_ask,213^v2^t3_esquery_v3&utm_term=p-tuning v2&spm=1018.2226.3001.4187)

[(109条消息) 知识图谱：【知识图谱问答KBQA（七）】——P-tuning V2训练代码核心网络层解析_J_Xiong0117的博客-CSDN博客](https://blog.csdn.net/u013010473/article/details/123051118?ops_request_misc=%7B%22request%5Fid%22%3A%22166840339316782429784930%22%2C%22scm%22%3A%2220140713.130102334.pc%5Fblog.%22%7D&request_id=166840339316782429784930&biz_id=0&utm_medium=distribute.pc_search_result.none-task-blog-2~blog~first_rank_ecpm_v1~rank_v31_ecpm-2-123051118-null-null.article_score_rank_blog&utm_term=p-tuning v2代码&spm=1018.2226.3001.4450)

[(109条消息) 知识图谱：【知识图谱问答KBQA（三）】——Prompt Learning_J_Xiong0117的博客-CSDN博客_prompt 知识图谱](https://blog.csdn.net/u013010473/article/details/122688390)

[(103条消息) Prompt-Tuning🔥🔥🔥_华师数据学院·王嘉宁的博客-CSDN博客](https://blog.csdn.net/qq_36426650/category_11415866.html)

[记录一次对past_key_values用法的理解 - 知乎 (zhihu.com)](https://zhuanlan.zhihu.com/p/459305102)



[【Coding】Hugging Face BertModel中的一些类和参数说明整理（二） - 知乎 (zhihu.com)](https://zhuanlan.zhihu.com/p/487728932)

P-tuning与Fine-tuning的最大区别，就是用一个小模型，学到这个prefix，大模型pretrained LM参数保持frozen的状态，来完成任务。