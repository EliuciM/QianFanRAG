### 文件结构

    gpt_doc
    │  constants.py		# 变量数值的配置文件
    │  ingest.py			# 封装了对文件进行切分编码的一系列函数
    │  query_api.py		# 针对特定文档进行查询的API接口文件
    │  upload_api.py 	# 用户上传文件到服务器的API接口文件
    │  README.md
    │
    ├─faissdb           	# faissdb的向量数据库存储路径
    ├─pic               	# 项目示例文件
    ├─templates       # 用户上传文件的前端页面
    │      web.html
    │
    └─upload		# 用户上传文件的存储路径

### 项目介绍

基于QianFan相关接口以及faiss数据库实现基于文件进行知识问答的接口，其中[upload_api.py](upload_api.py)完成了文件的上传工作（支持.txt .docx .doc 和 .pdf），[query_api.py](query_api.py)完成了文件的切分、向量化编码以及入库在调用，核心函数位于[ingest.py](ingest.py)，其中对编码后的向量进行了持久化存储，路径位于faissdb，因此在对相同文件进行访问时会绕过向量化和入库的部分，与之对应的，在更新时需要完全删除原始路径下的文件夹.

### 环境配置

- python 3.8.17
- faiss-cpu == 1.7.4
- Flask == 2.3.2
- 解析 doc 需要安装 libreoffice (for centos: yum install -y libreoffice)

### 项目示例

项目中有两个关键的变量query以及fileIdentifier，后者为指定的文档目录，需要和上传文件时的保持一致，前者为针对指定目录的提问。这里可以把fileIdentifier替换为用户的ID或者某一用户提交的文件类型,faissdb文件夹下的为提前创建好的索引文件。

- 上传文件的前端平台界面
  ![上传文件的前端平台界面](./pic/upload_api.bmp)
- 使用 Postman 发送 POST 请求，传入两个关键词，sub_id 对应 上方传输的 fileIdentifier, 即知识库的文件，query 对应提出的问题
- 在浏览器中发送 GET 请求：127.0.0.1:19000/ask?sub_id=200&query=你好，对应关系同上
