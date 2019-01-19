# encoding=utf-8
import jieba

seg_list = jieba.cut("月结的确是一个好方式，即方便了我们客户，也增强了人们使用顺丰的忠诚度。但是，我想说:顺丰快递，仅仅如此还不够！因为，同城如果有二个地点、三个地点发快递的话，仅仅注册地才能使用月结服务。给我们造成了困扰。如果你是全心合意为客户服务的话，那么，你不觉得还有改进的地方吗？", cut_all=False)
print("精确模式：" + " ".join(seg_list))
# print(seg_list)