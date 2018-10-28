面试题  
  
[TOC]  
  
[//]:#[PART1](https://blog.csdn.net/u013382288/article/details/80417681)  
  
## 2. 你系统的学习过机器学习算法吗？  
略。  
  
  
7. 还有一些围绕着项目问的具体问题  
略。  
  
8. 参加过哪些活动？  
略。  
  
## 9. hive？spark？sql？ nlp？  
    1）Hive允许使用类SQL语句在hadoop集群上进行读、写、管理等操作  
    2）Spark是一种与hadoop相似的开源集群计算环境，将数据集缓存在分布式内存中的计算平台，每轮迭代不需要读取磁盘的IO操作，从而答复降低了单轮迭代时间  
    
## 11. 还问了数据库，spark，爬虫（简历中有）  
略。  
  
12. 具体案例分析，关于京东商城销售的  
  
略。  
## 13. Linux基本命令  
    1）目录操作：ls、cd、mkdir、find、locate、whereis等  
    2）文件操作：mv、cp、rm、touch、cat、more、less  
    3）权限操作：chmod+rwx421  
    4）账号操作：su、whoami、last、who、w、id、groups等  
    5）查看系统：history、top  
    6）关机重启：shutdown、reboot  
    7）vim操作：i、w、w!、q、q!、wq等  
  
## 14. NVL函数  
    1）是oracle的一个函数  
    2）NVL( string1, replace_with)，如果string1为NULL，则NVL函数返回replace_with的值，否则返回原来的值  

## 16. sql中null与''的区别  
    1）null表示空，用is null判断  
    2）''表示空字符串，用=''判断  
 
18. 手写SQL  
略。  
  
  
## 19. SQL的数据类型  
    1）字符串：char、varchar、text  
    2）二进制串：binary、varbinary  
    3）布尔类型：boolean  
    4）数值类型：integer、smallint、bigint、decimal、numeric、float、real、double  
    5）时间类型：date、time、timestamp、interval  
  
20. C的数据类型  
    1）基本类型：  
        a. 整数类型：char、unsigned char、signed char、int、unsigned int、short、unsigned short、long、unsigned long  
        b. 浮点类型：float、double、long double  
    2）void类型  
    3）指针类型  
    4）构造类型：数组、结构体struct、共用体union、枚举类型enum  

## 22. roc图  
    1）以真阳（TP）为横轴，假阳为纵轴（FP），按照样本预测为真的概率排序，绘制曲线  
    2）ROC曲线下的面积为AUC的值  
  
## 23. 查准率查全率  
    1）查准率：TP/(TP+FP)  
    2）查全率：TP/(TP+FN)  
  
## 25. 内连接与外连接的区别  
    1）内连接：左右表取匹配行  
    2）外连接：分为左连接、右连接和全连接  
  
## 26. 欧式距离  
    1）字段取值平方和取开根号  
    2）表示m维空间中两个点的真实距离  
  
## 27. 普通统计分析方法与机器学习的区别  
  
这里不清楚普通统计分析方法指的是什么。  
如果是简单的统计分析指标做预测，那模型的表达能力是落后于机器学习的。  
如果是指统计学方法，那么统计学关心的假设检验，机器学习关心的是建模，两者的评估不同。  
  
28. BOSS面：关于京东的想法，哪里人，什么学校，多大了，想在京东获得什么，你能为京东提供什么，关于转正的解释，工作内容，拿到offer  
  
略。  
  
29. 先问了一个项目，然后问了工作意向，对工作是怎么看待的  
略。  
  
30. 问了一点Java很基础的东西，像set、list啥的  
略。  
  
31. 感觉一二面的面试官比较在意你会不会hive、sql  
略。  
  
  
## 33. 只是岗位名称一样，我一面问的都是围绕海量数据的推荐系统，二面就十几分钟，都是自己再说……感觉凉的不能再凉了  
    1）基于内容  
    2）协同过滤  
    3）基于矩阵分解   
    4）基于图  
其它包括冷启动、评估方法等  
  
## 34. 项目写的是天池比赛,只是大概描述了一下,特征工程和模型的选择  
    1）数据预处理  
    2）时间特征处理（sin化等）  
    3）连续特征处理（分箱等）  
    4）类别特征处理（onehot等）  
    5）交叉特征  
    6）特征hash化  
    7）gbdt构造特征  
    8）tfidf等对文本（或类似文本）的特征处理  
    9）统计特征  
    10）embedding方法作用于样本  
    11）聚类、SVD、PCA等  
    12）NN抽取特征  
    13）自动编码机抽取特征  

  
37. 用滑动窗口是怎样构造特征的  
文本和图像数据中，设置窗口大小与滑动步长，以窗口为片段抽取特征。  
  
## 44. 有一份分析报告，周一已定好框架，周五给老板，因为种种原因没能按时完成，怎么办？  
略。  
  
[PART2]()  
  
## 1. 二叉树题目  
略  
  
2. 层序遍历算法题  
    1）由顶向下逐层访问  
    2）可以用队列存储树，每次打印根节点并将左右节点放进队列  
（参考：https://www.cnblogs.com/masterlibin/p/5911298.html）  
  
3. 图论中的最大团、连通分量，然后问图划分的算法  
略  
  
4. 如何判断社区活跃度（基于图），现在想着可能是根据连通分量吧  
略。  
  
## 5. 给定相邻两个节点的相似度，怎么计算该点到其它点的相似度  
    1）把这个问题看成多维尺度分析问题（MDS），那么实际上就是已知点之间的距离，构造一个空间Z，使得这个空间内点之间的距离尽可能保持接近。点在新空间Z中的向量化就是点的表示，然后点到点的距离就可以。  
（MDS求解参考：https://blog.csdn.net/u010705209/article/details/53518772?utm_source=itdadao&utm_medium=referral）  
    2）其它：已知节点间距离，将节点embedding。这里我不太懂，希望大家有思路的可以指点下，谢啦  
    3）上诉两个答案也可能是我没看懂题意，因为该题的上下文是做复杂网络相关的研究。那么可能是知道任意两个相邻节点的相似度，求非相邻节点的相似度。这里可以参考simRank算法，即两个点的邻域越相似（有很多相似邻居），那么两个点越相似。有点像pageRank，是一个迭代的定义。（参考：https://blog.csdn.net/luo123n/article/details/50212207）  
  
## 6. 给一堆学生的成绩，将相同学生的所有成绩求平均值并排序，让我用我熟悉的语言，我就用了python的字典+sorted，面试官说不准用sort，然后问会别的排序，我就说了冒泡排序，原理我说了，然后问我还知道其他排序，答堆排序（其实我之前这方面复习了很多），之后问我有没有实现过（这个问题简直就是我的死角，就是没实现过，所以才想找个实习练练啊）  
    1）python直接pandas下groupby studentID sort  
    2）实现排序算法  
  
## 7. 问了我机器学习熟悉的算法，答svm，讲一下原理  
    1）一种分类方法，找到一个分类的超平面，将正负例分离，并让分类间隔尽可能大  
    2）过程：  
        a. 线性svm：损失函数如下  
  
        b. 对偶学习问题  
        c. 核函数：为了实现非线性分类，可以将样本映射到高维平面，然后用超平面分割。为了减少高维平面计算内积的操作，可以用一些“偷吃步”的方法同时进行高维映射和内积计算，就是核函数。包括多项式核函数、高斯核函数和sigmoid核函数  
        d. soft kernel  
（参考林轩田《机器学习技法》，SVM这部分的推导讲得很清楚；或者参考https://blog.csdn.net/abcjennifer/article/details/7849812/）  
  
    3）优点：  
        a. 容易抓住特征和目标之间的非线性关系  
        b. 避免陷入局部解，泛化能力强  
        c. 可以解决小样本高维问题（如文本分类）  
        d. 分类时只用到了支持向量，泛化能力强  
    4）缺点：  
        a. 训练时的计算复杂度高  
        b. 核函数选择没有通用方案  
        c. 对缺失数据敏感  
   
## 8. c中struct的对齐，我这个真的没听过，面试官让我之后自己查  
    为了提高存储器的访问效率，避免读一个成员数据访问多次存储器，操作系统对基本数据类型的合法地址做了限制，要求某种类型对象的地址必须是某个值K的整数倍（K=2或4或8）  
    1）Windows给出的对齐要求是:任何K（K=2或4或8）字节的基本对象的地址都必须是K的整数倍  
    2）Linux的对齐要求是：2字节类型的数据（如short）的起始地址必须是2的整数倍，而较大（int \*,int double ,long）的数据类型的地址必须是4的整数倍  
（参考：https://www.cnblogs.com/fengfenggirl/p/struct_align.html）  
  
## 9. 机器学习被调数据分析了，因为做推荐的，所以面试一直在聊具体场景的推荐方法，其他方面知识没有怎么问  
略。  
    
  
## 14. 50亿个url，找出指定的一个url  
50亿个的话是哈希查找，考虑到数量比较大会有冲突问题，那么可以用布隆过滤器。缺点还是会有误判，把不属于该集合的认为属于。  
  
  
17. 腾讯视频和优酷的区别  
略。  
  
  
## 19. KMP算法  
    1）目标是做字符串匹配，将时间复杂度从O(m\*n)降为O(m+n)  
    2）思想：利用了目标字符串内部的重复性，使比较时实现最大的移动量  
    3）方法：  
       a. 计算next[i]：表示字符串第1至i-1个字符的子串中，前缀后缀最长重复个数  
       b. 对比主串s和目标字符串ptr。当相等时，i和j都+1；当不相等时，j更新为next[j]。  
（参考：https://www.zhihu.com/question/21923021 @逍遥行 的回答）  
  
20. 哈夫曼编码  
    1）一种编码方式，让出现次数越多的字符编码越短，从而压缩编码的长度  
    2）过程：  
        a. 建立一个加权树，出现次数越多的字符出现在越上层  
        b. 左连接为0，右连接为1  
        c. 到达字符的01组合记为该字符的编码  
        d. 由于哈夫曼编码是前缀编码（如果没有一个编码是另一个编码的前缀，则称这样的编码为前缀编码。如0,101和100是前缀编码），因此可以唯一地还原  
（参考：https://blog.csdn.net/xgf415/article/details/52628073，https://www.cnblogs.com/xidongyu/p/6031518.html）  
  
## 21. 给出一个商业业务例子，这个例子中使用模型会比数据查询和简单的统计分析更有效果  
    1）推荐算法  
    2）异常值检测  
    3）精准营销  
    4）信贷发放  
    ......  
  
22. 了不了解mapreduce  
略。  
  
## 23. 数据库熟练程度  
略。  
  
## 24. 平时看什么书  
略。  
  
## 25. 偏差和方差  
    1）偏差：预测值与真实值差异，偏差大表示欠拟合。然后引申到计算方式和解决方法  
    2）方差：预测值与均值的波动，方差大表示过拟合。然后引申到计算方式和解决方法  
  
## 27. 一个线段上任意取两点，能组成三角形的概率  
    1）设线段长度为1，三个子线段长度为x1,x2,x3  
    2）根据三角形两边之和大于第三边，可得：  
            a. x1+x2>x3  
            b. x1-x2<x3  
            c. x2-x1<x3  
            d. x1+x2+x3=1  
    3）将x3=1-x1-x2带入abc，然后用x1、x2为轴绘制，可以得到有效面积为1/8  
  
## 28. 有uid，app名称，app类别，数据百亿级别，设计算法算出每个app类别只安装了一个app的uid总数。  
    应该用map reduce吧，但我不会啊。准备写个sql，结果写了半天还是写不出。面试完走到楼下就想出来了，233  
    1）小数据量的话直接查询：  
        select count(*) as result from  
        (select uid, category, count(app) as c from table  
        group by category having c = 1) as t  
        group by t.uid having count(category) = 类别数  
    2）大数据量下（没用过hadoop不太清楚，望大家指正）  
        a. 原始文件可以拼接为uid-app-categroy  
        b. map阶段形成的<k,v>是<uid-category,1>  
        c. reduce阶段统计key为“uid-category”的count数量  
        d. 只保留count为1的数据  
        e. 剩下的数据量直接统计uid出现次数=category类别数的数据  
  
## 29. 有一个网页访问的数据，包含uid，ip地址，url，文章资料。设计算法预测用户性别  
    1）分类问题用机器学习方法解（这里假设已经有部分用户的性别标签）  
    2）想到的特征有：  
        a. 文档主题词（top3）  
  b. 文档标题词（按照标题词在文档中出现的频率，取top3）（参考：https://blog.csdn.net/u013382288/article/details/80385814）   
        c. ip地址  
        d. url中参数（如网页搜索中的query）  
        e. 统计特征：访问数量、单页面平局访问时间、上网时间等  
  
笔试题  
  
1、对于过拟合有什么方法处理  
    见上文。  
  
2、冒泡排序  
    1）时间复杂度O（n）到O（n²）  
    2）稳定排序  
  
3、排列组合  
    略。  
  
4、大数定律和切比雪夫不等式的式子  
    方差越大，X落在区间外的概率越大，X的波动也就越大。  
  
5、回归系数的计算  
    略。  
  
6、鞍点的Hessian矩阵是否正定  
    半正定，含有0特征值。  
  
7、快速排序的最佳状况  
    O（nlogn）  
  
8、对于svm，梯度消失怎么在图像上判定  
    不懂。  
  
9、超参不敏感  
    见下文，有详细题目。  
  
10、分层抽样的适用范围  
    略。  
  
11、贝叶斯公式  
    略。  
  
12、高数里的一些求导的知识  
    略。  
  
13、线性代数里的秩、克莱姆法则  
    1）向量组中的秩，就是极大线性无关向量组中的向量个数  
    2）我们可以认为一个矩阵的秩，是给矩阵按质量排序的依据。  
    秩越高的矩阵内容越丰富，冗余信息越少。秩越低的矩阵废数据越多。  
（知乎 @苗加加）  
  
3）克莱姆法则是求解线性方程组的定理，详见：https://baike.baidu.com/item/%E5%85%8B%E8%8E%B1%E5%A7%86%E6%B3%95%E5%88%99/7211518?fr=aladdin；https://blog.csdn.net/yjf_victor/article/details/45065847  
  
14、推导回归系数的过程  
（参考：https://blog.csdn.net/marsjohn/article/details/54911788）  
  
15、深度优先遍历  
    1）图的深度优先遍历：  
        a. 首先以一个未被访问过的顶点作为起始顶点，沿当前顶点的边走到未访问过的顶点；  
        b. 当没有未访问过的顶点时，则回到上一个顶点，继续试探别的顶点，直到所有的顶点都被访问过  
    2）二叉树的深度优先遍历：实际就是前序遍历  
  
  
解答题：  
  
1、解释机器学习中的偏差和方差，对不同的情况应该采取什么样的措施？  
  
    见上文。  
  
2、描述假设检验的过程  
    1）设置原假设H0，备择假设H1（一般我们的研究假设是H1）  
    2）选定检验方法  
    3）计算观测到的数值分分布，如果实际观察发生的是小概率事件，并且超过显著性水平，那么认为可以排除原假设H0  
  
3、
  
笔试题  
  
1.深度学习，训练集误差不断变小，测试集误差变大，要怎么做（ACD）  
A 数据增强 B 增加网络深度 C提前停止训练 D增加 dropout  
  
2. 鞍点的Hessian矩阵是？  
半正定。  
  
3.快排的时间复杂度  
O（nlogn）  
  
4 哪个sigmoid函数梯度消失最快？是零点处导数最大的还是最小的？  
零点处导数最大。  
  
5. 5 7 0 9 2 3 1 4 做冒泡排序的交换次数？  
16？  
  
6. 哪种优化方法对超参数不敏感？（C）  
SGD BGD Adadelta Momentum  
  
1）SGD受到学习率α影响  
  
2）BGD受到batch规模m影响  
  
3）Adagrad的一大优势时可以避免手动调节学习率，比如设置初始的缺省学习率为0.01，然后就不管它，另其在学习的过程中自己变化。  
  
为了避免削弱单调猛烈下降的减少学习率，Adadelta产生了1。Adadelta限制把历史梯度累积窗口限制到固定的尺寸w，而不是累加所有的梯度平方和  
  
4）Momentum：也受到学习率α的影响  
[二](https://blog.csdn.net/u013382288/article/details/80470316)  
  
## 三  
1、海量日志数据，提取出某日访问百度次数最多的那个IP。  
　　首先是这一天，并且是访问百度的日志中的IP取出来，逐个写入到一个大文件中。注意到IP是32位的，最多有个2^32个IP。同样可以采用映射的方法，比如模1000，把整个大文件映射为1000个小文件，再找出每个小文中出现频率最大的IP（可以采用hash_map进行频率统计，然后再找出频率最大的几个）及相应的频率。然后再在这1000个最大的IP中，找出那个频率最大的IP，即为所求。  
　　或者如下阐述：  
　　算法思想：分而治之+Hash  
1.IP地址最多有2^32=4G种取值情况，所以不能完全加载到内存中处理；  
2.可以考虑采用“分而治之”的思想，按照IP地址的Hash(IP)24值，把海量IP日志分别存储到1024个小文件中。这样，每个小文件最多包含4MB个IP地址；  
3.对于每一个小文件，可以构建一个IP为key，出现次数为＆#118alue的Hash  
 map，同时记录当前出现次数最多的那个IP地址；  
4.可以得到1024个小文件中的出现次数最多的IP，再依据常规的排序算法得到总体上出现次数最多的IP；  
2、搜索引擎会通过日志文件把用户每次检索使用的所有检索串都记录下来，每个查询串的长度为1-255字节。  
　　假设目前有一千万个记录（这些查询串的重复度比较高，虽然总数是1千万，但如果除去重复后，不超过3百万个。一个查询串的重复度越高，说明查询它的用户越多，也就是越热门。），请你统计最热门的10个查询串，要求使用的内存不能超过1G。  
　　典型的Top K算法，还是在这篇文章里头有所阐述，  
　　文中，给出的最终算法是：  
　　第一步、先对这批海量数据预处理，在O（N）的时间内用Hash表完成统计（之前写成了排序，特此订正。July、2011.04.27）；  
　　第二步、借助堆这个数据结构，找出Top K，时间复杂度为N‘logK。  
　　即，借助堆结构，我们可以在log量级的时间内查找和调整/移动。因此，维护一个K(该题目中是10)大小的小根堆，然后遍历300万的Query，分别和根元素进行对比所以，我们最终的时间复杂度是：O（N）  
 + N’*O（logK），（N为1000万，N’为300万）。ok，更多，详情，请参考原文。  
　　或者：采用trie树，关键字域存该查询串出现的次数，没有出现为0。最后用10个元素的最小推来对出现频率进行排序。  
3、有一个1G大小的一个文件，里面每一行是一个词，词的大小不超过16字节，内存限制大小是1M。返回频数最高的100个词。  
　　方案：顺序读文件中，对于每个词x，取hash(x)P00，然后按照该值存到5000个小文件（记为x0,x1,…x4999）中。这样每个文件大概是200k左右。  
　　如果其中的有的文件超过了1M大小，还可以按照类似的方法继续往下分，直到分解得到的小文件的大小都不超过1M。  
　　对每个小文件，统计每个文件中出现的词以及相应的频率（可以采用trie树/hash_map等），并取出出现频率最大的100个词（可以用含100个结点的最小堆），并把100个词及相应的频率存入文件，这样又得到了5000个文件。下一步就是把这5000个文件进行归并（类似与归并排序）的过程了。  
4、有10个文件，每个文件1G，每个文件的每一行存放的都是用户的query，每个文件的query都可能重复。要求你按照query的频度排序。  
　　还是典型的TOP K算法，解决方案如下：  
　　方案1：  
　　顺序读取10个文件，按照hash(query)的结果将query写入到另外10个文件（记为）中。这样新生成的文件每个的大小大约也1G（假设hash函数是随机的）。  
　　找一台内存在2G左右的机器，依次对用hash_map(query,query_count)来统计每个query出现的次数。利用快速/堆/归并排序按照出现次数进行排序。将排序好的query和对应的query_cout输出到文件中。这样得到了10个排好序的文件（记为）。  
　　对这10个文件进行归并排序（内排序与外排序相结合）。  
　　方案2：  
　　一般query的总量是有限的，只是重复的次数比较多而已，可能对于所有的query，一次性就可以加入到内存了。这样，我们就可以采用trie树/hash_map等直接来统计每个query出现的次数，然后按出现次数做快速/堆/归并排序就可以了。  
　　方案3：  
　　与方案1类似，但在做完hash，分成多个文件后，可以交给多个文件来处理，采用分布式的架构来处理（比如MapReduce），最后再进行合并。  
5、  
给定a、b两个文件，各存放50亿个url，每个url各占64字节，内存限制是4G，让你找出a、b文件共同的url？  
　　方案1：可以估计每个文件安的大小为5G×64=320G，远远大于内存限制的4G。所以不可能将其完全加载到内存中处理。考虑采取分而治之的方法。  
　　遍历文件a，对每个url求取hash(url)00，然后根据所取得的值将url分别存储到1000个小文件（记为a0,a1,…,a999）中。这样每个小文件的大约为300M。  
　　遍历文件b，采取和a相同的方式将url分别存储到1000小文件（记为b0,b1,…,b999）。这样处理后，所有可能相同的url都在对应的小文件（a0vsb0,a1vsb1,…,a999vsb999）中，不对应的小文件不可能有相同的url。然后我们只要求出1000对小文件中相同的url即可。  
　　求每对小文件中相同的url时，可以把其中一个小文件的url存储到hash_set中。然后遍历另一个小文件的每个url，看其是否在刚才构建的hash_set中，如果是，那么就是共同的url，存到文件里面就可以了。  
　　方案2：如果允许有一定的错误率，可以使用Bloom filter，4G内存大概可以表示340亿bit。将其中一个文件中的url使用Bloom  
 filter映射为这340亿bit，然后挨个读取另外一个文件的url，检查是否与Bloomfilter，如果是，那么该url应该是共同的url（注意会有一定的错误率）。  
Bloom filter日后会在本BLOG内详细阐述。  
6、在2.5亿个整数中找出不重复的整数，注，内存不足以容纳这2.5亿个整数。  
　　方案1：采用2-Bitmap（每个数分配2bit，00表示不存在，01表示出现一次，10表示多次，11无意义）进行，共需内存2^32  
 * 2 bit=1 GB内存，还可以接受。然后扫描这2.5亿个整数，查看Bitmap中相对应位，如果是00变01，01变10，10保持不变。所描完事后，查看bitmap，把对应位是01的整数输出即可。  
　　方案2：也可采用与第1题类似的方法，进行划分小文件的方法。然后在小文件中找出不重复的整数，并排序。然后再进行归并，注意去除重复的元素。  
7、腾讯面试题：给40亿个不重复的unsigned int的整数，没排过序的，然后再给一个数，如何快速判断这个数是否在那40亿个数当中？  
　　与上第6题类似，我的第一反应时快速排序+二分查找。以下是其它更好的方法：  
　　方案1：oo，申请512M的内存，一个bit位代表一个unsigned  
 int值。读入40亿个数，设置相应的bit位，读入要查询的数，查看相应bit位是否为1，为1表示存在，为0表示不存在。  
　　方案2：这个问题在《编程珠玑》里有很好的描述，大家可以参考下面的思路，探讨一下：  
　　又因为2^32为40亿多，所以给定一个数可能在，也可能不在其中；  
　　这里我们把40亿个数中的每一个用32位的二进制来表示  
　　假设这40亿个数开始放在一个文件中。  
　　然后将这40亿个数分成两类:  
1.最高位为0  
2.最高位为1  
　　并将这两类分别写入到两个文件中，其中一个文件中数的个数<=20亿，而另一个>=20亿（这相当于折半了）；  
　　与要查找的数的最高位比较并接着进入相应的文件再查找  
　　再然后把这个文件为又分成两类:  
1.次最高位为0  
2.次最高位为1  
　　并将这两类分别写入到两个文件中，其中一个文件中数的个数<=10亿，而另一个>=10亿（这相当于折半了）；  
　　与要查找的数的次最高位比较并接着进入相应的文件再查找。  
　　…….  
　　以此类推，就可以找到了,而且时间复杂度为O(logn)，方案2完。  
　　附：这里，再简单介绍下，位图方法：  
　　使用位图法判断整形数组是否存在重复  
　　判断集合中存在重复是常见编程任务之一，当集合中数据量比较大时我们通常希望少进行几次扫描，这时双重循环法就不可取了。  
　　位图法比较适合于这种情况，它的做法是按照集合中最大元素max创建一个长度为max+1的新数组，然后再次扫描原数组，遇到几就给新数组的第几位置上1，如遇到5就给新数组的第六个元素置1，这样下次再遇到5想置位时发现新数组的第六个元素已经是1了，这说明这次的数据肯定和以前的数据存在着重复。这种给新数组初始化时置零其后置一的做法类似于位图的处理方法故称位图法。它的运算次数最坏的情况为2N。如果已知数组的最大值即能事先给新数组定长的话效率还能提高一倍。  
　　欢迎，有更好的思路，或方法，共同交流。  
8、怎么在海量数据中找出重复次数最多的一个？  
　　方案1：先做hash，然后求模映射为小文件，求出每个小文件中重复次数最多的一个，并记录重复次数。然后找出上一步求出的数据中重复次数最多的一个就是所求（具体参考前面的题）。  
9、上千万或上亿数据（有重复），统计其中出现次数最多的钱N个数据。  
　　方案1：上千万或上亿的数据，现在的机器的内存应该能存下。所以考虑采用hash_map/搜索二叉树/红黑树等来进行统计次数。然后就是取出前N个出现次数最多的数据了，可以用第2题提到的堆机制完成。  
10、一个文本文件，大约有一万行，每行一个词，要求统计出其中最频繁出现的前10个词，请给出思想，给出时间复杂度分析。  
　　方案1：这题是考虑时间效率。用trie树统计每个词出现的次数，时间复杂度是O(n*le)（le表示单词的平准长度）。然后是找出出现最频繁的前10个词，可以用堆来实现，前面的题中已经讲到了，时间复杂度是O(n*lg10)。所以总的时间复杂度，是O(n*le)与O(n*lg10)中较大的哪一个。  
　　附、100w个数中找出最大的100个数。  
　　方案1：在前面的题中，我们已经提到了，用一个含100个元素的最小堆完成。复杂度为O(100w*lg100)。  
　　方案2：采用快速排序的思想，每次分割之后只考虑比轴大的一部分，知道比轴大的一部分在比100多的时候，采用传统排序算法排序，取前100个。复杂度为O(100w*100)。  
　　方案3：采用局部淘汰法。选取前100个元素，并排序，记为序列L。然后一次扫描剩余的元素x，与排好序的100个元素中最小的元素比，如果比这个最小的要大，那么把这个最小的元素删除，并把x利用插入排序的思想，插入到序列L中。依次循环，知道扫描了所有的元素。复杂度为O(100w*100)。  
[三](https://blog.csdn.net/smarthhl/article/details/50390321)  
  
#四  
  
## 3.以下算法对缺失值敏感的模型包括：（AE）  
  
A、Logistic Regression（逻辑回归）  
B、随机森林  
C、朴素贝叶斯  
D、C4.5  
E、SVM  
  
逻辑回归（目标变量是二元变量）  
  
建模数据量不能太少，目标变量中每个类别所对应的样本数量要足够充分，才能支持建模  
排除共线性问题（自变量间相关性很大）  
异常值会给模型带来很大干扰，要剔除。  
 逻辑回归不能处理缺失值，所以之前应对缺失值进行适当处理。  
  
随机森林的优点：  
    可以处理高维数据，不同进行特征选择（特征子集是随机选择）  
    模型的泛化能力较强  
    训练模型时速度快，成并行化方式，即树之间相互独立  
    模型可以处理不平衡数据，平衡误差  
    最终训练结果，可以对特种额排序，选择比较重要的特征  
    随机森林有袋外数据（OOB），因此不需要单独划分交叉验证集  
    对缺失值、异常值不敏感  
    模型训练结果准确度高  
    相对Bagging能够收敛于更小的泛化误差  
  
朴素贝叶斯的假设前提有两个第一个为：各特征彼此独立；第二个为且对被解释变量的影响一致，不能进行变量筛选  
朴素贝叶斯对缺失值不敏感它  

SVM：  
最优分类面就是要求分类线不但能将两类正确分开(训练错误率为0),且使分类间隔最大。SVM考虑寻找一个满足分类要求的超平面,并且使训练集中的点距离分类面尽可能的远,也就是寻找一个分类面使它两侧的空白区域(margin)最大。  
C是惩罚因子，是一个由用户去指定的系数，表示对分错的点加入多少的惩罚，当C很大的时候，分错的点就会更少，但是过拟合的情况可能会比较严重，当C很小的时候，分错的点可能会很多，不过可能由此得到的模型也会不太正确。  
SVM的优点：  
可以解决小样本，高维和非线性问题。  
可以避免神经网络结构选择和局部极小点问题。  
SVM的缺点：  
对缺失数据敏感。  
对非线性问题没有通用解决方案，须谨慎选择不同Kernelfunction来处理。  
  

[四](https://blog.csdn.net/nilhurui/article/details/81346332)  
  
#五  
## 1. 贝叶斯公式复述并解释应用场景  
   1）P（A|B) = P(B|A)\*P(A) / P(B)  
   2）如搜索query纠错，设A为正确的词，B为输入的词，那么：  
      a. P(A|B)表示输入词B实际为A的概率  
      b. P(B|A)表示词A错输为B的概率，可以根据AB的相似度计算（如编辑距离）  
      c. P(A)是词A出现的频率，统计获得  
      d. P(B)对于所有候选的A都一样，所以可以省去  
  
## 2. 如何写SQL求出中位数平均数和众数（除了用count之外的方法）  
   1）中位数：  
方案1（没考虑到偶数个数的情况）：  
set @m = (select count(\*)/2 from table)  
select column from table order by column limit @m, 1  
  
方案2（考虑偶数个数，中位数是中间两个数的平均）：  
set @index = -1  
select avg(table.column)  
from  
(select @index:=@index+1 as index, column  
from table order by column) as t  
where t.index in (floor(@index/2),ceiling(@index/2))  
   2）平均数：select avg(distinct column) from table  
   3）众数：select column, count(\*) from table group by column order by column desc limit 1(emmm，好像用到count了）  
  
3. 学过的机器学习算法有哪些  
略。  
  
8. 对拼多多有什么了解，为什么选择拼多多  
略。  
  
## 9. 口答两个SQL题（一个跟留存率相关，一个要用到row number）  
   1）留存率：略  
   2）mysql中设置row number：  
SET @row_number = 0; SELECT (@row_number:=@row_number + 1) AS num FROM table  
  
12. 为什么选择拼多多  
略。  
  
13. 用过拼多多下单没，感受如何  
略。  
  
14. 可以接受单休和加班么  
略。  
  
15. 为啥要选数据分析方向（我简历上写的是数据挖掘工程师。。。）  
略。  
  
16. 开始聊项目，深究项目，我研究生阶段的方向比较偏，所以面试的三分之二时间都是在给他讲项目，好在最后他终于听懂了，thx god、、、  
略。  
  
17. hadoop原理和mapreduce原理  
   1）Hadoop原理：采用HDFS分布式存储文件，MapReduce分解计算，其它先略  
   2）MapReduce原理：  
      a. map阶段：读取HDFS中的文件，解析成<k,v>的形式，并对<k,v>进行分区（默认一个区），将相同k的value放在一个集合中  
      b. reduce阶段：将map的输出copy到不同的reduce节点上，节点对map的输出进行合并、排序  
（参考：https://www.cnblogs.com/ahu-lichang/p/6645074.html）  
  
18. 还有啥问题要问的？于是我出于本能的问了一句“为啥不写代码！” 然后面试官说“时间不够了。。。。”  
略。  
  
## 19.现有一个数据库表Tourists，记录了某个景点7月份每天来访游客的数量如下： id date visits 1 2017-07-01 100 …… 非常巧，id字段刚好等于日期里面的几号。现在请筛选出连续三天都有大于100天的日期。 上面例子的输出为： date 2017-07-01 ……  
  
解：  
select t1.date  
from Tourists as t1, Tourists as t2, Tourists as t3  
on t1.id = (t2.id+1) and t2.id = (t3.id+1)  
where t1.visits >100 and t2.visits>100 and t3.visits>100  
  
## 20.在一张工资表salary里面，发现2017-07这个月的性别字段男m和女f写反了，请用一个Updae语句修复数据 例如表格数据是： id name gender salary month 1 A m 1000 2017-06 2 B f 1010 2017-06  
解：  
update salary  
set gender = replace('mf', gender, '')  
  
## 21.现有A表，有21个列，第一列id，剩余列为特征字段，列名从d1-d20，共10W条数据！ 另外一个表B称为模式表，和A表结构一样，共5W条数据 请找到A表中的特征符合B表中模式的数据，并记录下相对应的id 有两种情况满足要求： 1 每个特征列都完全匹配的情况下。 2 最多有一个特征列不匹配，其他19个特征列都完全匹配，但哪个列不匹配未知  
解：（这题不懂怎么解）  
select A.id,  
((case A.d1 when B.d1 then 1 else 0) +  
(case A.d2 when B.d2 then 1 else 0) +  
...) as count_match  
from A left join B  
on A.d1 = B.d1  
  
## 22.我们把用户对商品的评分用稀疏向量表示，保存在数据库表t里面： t的字段有：uid，goods_id，star uid是用户id；goodsid是商品id；star是用户对该商品的评分，值为1-5。 现在我们想要计算向量两两之间的内积，内积在这里的语义为：对于两个不同的用户，如果他们都对同样的一批商品打了分，那么对于这里面的每个人的分数乘起来，并对这些乘积求和。 例子，数据库表里有以下的数据： U0 g0 2 U0 g1 4 U1 g0 3 U1 g1 1 计算后的结果为： U0 U1 2*3+4*1=10 ……  
解：  
select uid1, uid2, sum(result) as dot  
from  
(select t1.uid as uid1, t2.uid as uid2, t1.goods_id, t1.star\*t2.star as result  
from t as t1, t as t2  
on t1.goods_id = t2.goods_id) as t  
group by goods_id  
  
23.微信取消关注分析，题目太长了，没记录  
略。  
  
## 24. 统计教授多门课老师数量并输出每位老师教授课程数统计表  
解：设表class中字段为id，teacher，course  
  
1）统计教授多门课老师数量  
select count(*) from class  
group by teacher having count(*) > 1  
  
2）输出每位老师教授课程数统计  
select teacher, count(course) as count_course  
from class  
group by teacher  
  
## 25. 四个人选举出一个骑士，统计投票数，并输出真正的骑士名字  
解：设表tabe中字段为id，knight，vote_knight  
select knight from table  
group by vote_knight  
order by count(vote_knight) limit 1  
  
## 26. 员工表，宿舍表，部门表，统计出宿舍楼各部门人数表  
解：设员工表为employee，字段为id，employee_name，belong_dormitory_id，belong_department_id；  
宿舍表为dormitory，字段为id，dormitory_number；  
部门表为department，字段为id，department_name  
select dormitory_number, department_name, count(employee_name) as count_employee  
from employee as e  
left join dormitory as dor on e.belong_dormitory_id = dor.id  
left join department as dep on e.belong_department_id = dep.id  
  
## 27. 给出一堆数和频数的表格，统计这一堆数中位数  
解：设表table中字段为id,number,frequency  
set @sum = (select sum(frequency)+1 as sum from table)  
set @index = 0  
set @last_index = 0  
select avg(distinct t.frequecy)  
from  
(select @last_index := @index, @index := @index+frequency as index, frequency  
from table) as t  
where t.index in (floor(@sum/2), ceiling(@sum/2))  
or (floor(@sum/2) > t.last_index and ceiling(@sum.2) <= t.index)  
  
## 28. 中位数，三个班级合在一起的一张成绩单，统计每个班级成绩中位数  
解：设表table中字段为id，class，score  
select t1.class, avg(distinct t1.score) as median  
from table t1, table t2 on t1.id = t2.id  
group by t1.class, t1.score  
having sum(case when t1.score >= t2.score then 1else 0 end) >=  
(select count(*)/2 from table where table.class = t1.class)  
and  
having sum(case when t1.score <= t2.score then 1else 0 end) >=  
(select count(*)/2 from table where table.class = t1.class)  
  
## 29. 交易表结构为user_id,order_id,pay_time,order_amount  
   写sql查询过去一个月付款用户量（提示 用户量需去重）最高的3天分别是哪几天  
  写sql查询做昨天每个用户最后付款的订单ID及金额  
  
1）select count(distinct user_id) as c from table group by month(pay_time) order by c desc limit 3  
  
2）select order_id, order_amount from ((select user_id, max(pay_time) as mt from table group by user_id where DATEDIFF(pay_time, NOW()) = -1 as t1) left join table as t2 where t1.user_id = t2.user_id and t1.mt == t2.pay_time)  
  
   
  
## 30. PV表a(表结构为user_id,goods_id),点击表b(user_id,goods_id),数据量各为50万条，在防止数据倾斜的情况下，写一句sql找出两个表共同的user_id和相应的goods_id  
  
select * from a  
where a.user_id exsit (select user_id from b)  
（这题不太懂，sql中如何防止数据倾斜）  
  
## 31. 表结构为user_id,reg_time,age, 写一句sql按user_id随机抽样2000个用户  写一句sql取出按各年龄段（每10岁一个分段，如（0,10））分别抽样1%的用户  
  
1）随机抽样2000个用户  
select * from table order by rand() limit 2000  
  
2）取出各年龄段抽样1%的用户  
set @target = 0  
set @count_user = 0  
select @target:=@target+10 as age_right, *  
from table as t1  
where t1.age >=@target-10 and t1.age < (@target)  
and t1.id in  
(select floor(count(*)*0.1） from table as t2  
where t1.age >=@target-10 and t1.age < (@target)  
order by rand() limit ??)  
  
（mysql下按百分比取数没有想到比较好的方法，因为limit后面不能接变量。想到的方法是先计算出每个年龄段的总数，然后计算出1%是多少，接着给每一行加一个递增+1的行标，当行标=1%时，结束）  
  
## 32. 用户登录日志表为user_id,log_id,session_id,plat,visit_date 用sql查询近30天每天平均登录用户数量  用sql查询出近30天连续访问7天以上的用户数量  
  
1）近三十天每天平均登录用户数量  
select visit_date, count(distince user_id)  
group by visit_date  
  
2）近30天连续访问7天以上的用户数量  
  
select t1.date  
from table t1, table t2, ..., table t7  
on t1.visit_date = (t2.visit_date+1) and t2.visit_date = (t3.visit_date+1)  
and ... and t6.visit_date = (t7.visit_date+1）  
  
## 33. 表user_id,visit_date,page_name,plat  统计近7天每天到访的新用户数 统计每个访问渠道plat7天前的新用户的3日留存率和7日留存率  
  
1）近7天每天到访的新用户数  
select day(visit_date), count(distinct user_id)  
from table  
where user_id not in  
(select user_id from table  
where day(visit_date) < date_sub(visit_date, interval 7day))  
  
2）每个渠道7天前用户的3日留存和7日留存  
 三日留存  
  
 先计算每个平台7日前的新用户数量  
select t1.plat, t1.c/t2.c as retention_3  
(select plat, count(distinct user_id)  
from table  
group by plat, user_id  
having day(min(visit_date)) = date_sub(now(), interval 7 day)) as t1  
left join  
(select plat, count(distinct user_id) as c  
from table  
group by user_id having count(user_id) > 0  
having day(min(visit_date)) = date_sub(now(), interval 7 day)  
and day(max(visit_date)) > date_sub(now(), interval 7 day)  
and day(max(visit_date)) <= date_sub(now(), interval 4day)) as t2  
on t1.plat = t2.plat  
[五](https://blog.csdn.net/u013382288/article/details/80450360)  
  
#六[source](https://blog.csdn.net/Data_learning/article/details/81434426)  
 
## 七  
## 1. 做自我介绍，着重介绍跟数据分析相关的经验，还有自己为什么要做数据分析  
略。  
  
## 3. 关于假设检验的问题，然而我并没有答上来，面试官说没关系  
假设检验的基本原理是：全称命题不能证明但可以被证伪。  
令我们研究假设的相反假设为原假设，认为我们研究假设的发生是小概率事件。  
如果我们的观察值是研究假设，那么认为可以排除原假设，我们的研究假设并不是小概率事件。  
  
4. 问了笔试中的题目为什么没做，现场做  
略。  
  
5. 对今日头条的看法  
略。  
  
6. 关于采样的问题  
略。  
  
  
9. 最后问头条的使用感受  
略。  
  
10. 为什么做数据分析  
略。  
  
11. 自己的优缺点  
略。  
    
## 15. 立方体每面抽掉一层非棱角部分，面积和体积的变化  
看不懂题意。  
  
## 18. 如何学习新知识? (思路大概就是利用什么渠道，怎么获取)  
略  
  
## 19. 行存储和列存储的区别  
    1）行存储：传统数据库的存储方式，同一张表内的数据放在一起，插入更新很快。缺点是每次查询即使只涉及几列，也要把所有数据读取  
    2）列存储：OLAP等情况下，将数据按照列存储会更高效，每一列都可以成为索引，投影很高效。缺点是查询是选择完成时，需要对选择的列进行重新组装。  
“当你的核心业务是 OLTP 时，一个行式数据库，再加上优化操作，可能是个最好的选择。  
当你的核心业务是 OLAP 时，一个列式数据库，绝对是更好的选择”  
  
（参考：https://blog.csdn.net/qq_26091271/article/details/51778675；https://www.zhihu.com/question/29380943）  
[七](https://blog.csdn.net/u013382288/article/details/80390324)  
  
## 八  

