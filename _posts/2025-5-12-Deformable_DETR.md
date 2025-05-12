#### DEFORMABLE DETR: DEFORMABLE TRANSFORMERS  FOR END-TO-END OBJECT DETECTION（可变形变压器）

---

**对比DETR的优势**

- 相比于DETR能更短的**训练周期**就可以达到收敛
- 对于检测小目标的性能优于DETR，因为DETR对于高分辨率特征图会导致计算复杂度难以处理，对特征图的**计算和内存复杂度都极高**
- 当模型初始化时，自注意力模块会对特征图中的所有像素分配几乎均匀的注意力权重，需要经过长时间的训练，注意力权重才会学会聚焦于**稀疏且有意义的像素位置**---**可变形卷积**能很好的避免这个问题（关注稀疏的空间位置）

---

**Deformable DETR原理**

- 通过**Deformable conv可变形卷积**的稀疏空间采样优势和**Tramsformer**的全局关系建模能力相结合
- **Deformable attention module(可变形注意力模块)**通过仅关注少量采样位置，作为从所有特征图像素中筛选出关键元素的预过滤器；利用注意力模块能自然地聚合多尺度特征从而**无需借助特征金字塔**

---

**解决Transformer计算量和内存复杂度高的问题**

- 使用预定义的稀疏注意力模式对于关键元素进行处理。将注意力模式限制为固定的局部窗口。
- 学习数据依赖的稀疏注意力---哈希，k-means，块排列
- 改进self-attention中的低秩特性

![image-20250512194816587](C:\Users\lwj\AppData\Roaming\Typora\typora-user-images\image-20250512194816587.png)

![image-20250512194827755](C:\Users\lwj\AppData\Roaming\Typora\typora-user-images\image-20250512194827755.png)