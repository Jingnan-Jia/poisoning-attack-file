# poisoning-attack-file
我会把poisoning-attack相关文件保存在这里。

这个文件夹是用来验证这么个东西：

lamda=3,30,300的时候，模型的success rate，average acc， per acc分别有什么影响？
我这里在做lamda=30实验，卢那里做３的实验。（因为我不知道他那里把original model存在哪里了，所以我给他的程序是重新跑一下original model，而我自己的这个程序是调用本地已经训练好的保存好的ｏｒｉｇｉｎａｌ　ｍｏｄｅｌ）

用lamda=30做实验。
。。。
算了吧，我发现代码写得有问题，卢的代码是好的他确认了没有问题，我不再恋战了，赶紧取看看怎么弄那个特征图吧！！！

tf_get_nn_of_x.py
这个文件是完整的寻找ｘ（代码里写的x=x_test[0]）的最近邻的代码，同时包括了save_fig, save_figs两个完整的保存单张图片和多张图片的函数。代码描述：寻找最近邻的时候，用白化后的数据求概率向量，与ｘ的概率向量比较，对这些距离排序后，得到前N个数据的索引值，再从白化之前的原始图片中找出这些索引值对应的图片。

