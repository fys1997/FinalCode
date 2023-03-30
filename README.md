# FinalCode
This is the pytorch code for my graduation paper of master's degree

此分支同来存放论文第三章Meta Learning中处理BJ Flow数据集的代码

2.12 证实原数据BJ_Flow (day,hour,row,col,2) reshape 为(day*hour,row*col,2)与原数据顺序一致吻合

2.13 完成generate_data.py 重写，生成train val test数据集 8:1:1

2.16 搞懂了ST-MetaNet的代码逻辑(其在计算attention时只计算了跟他直接相邻的点)以及数据加载方式

2.19 完成NMC.py与AdjLearner.py
