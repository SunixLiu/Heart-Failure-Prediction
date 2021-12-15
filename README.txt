心衰生存与风险预测模型演示说明
1.本目录包含以下主要文件
  运行于Jupyter Notebook的主文件
  Heart_Failure_Survival_Hazard_Model_Prediction.ipynb
  运行于命令行以启动Web端的Python文件
  HF_prediction_v1.py
以上两个文件均有代码注释说明
 rsf_all.pkl和 explainer_all.sav 分别是已经生成的模型权重和特征解释文件，是运行Web必须的文件
4.启动Web端
  .确保安装了streamlit (pip install -U streanlit)
  .执行 streamlit run HF_prediction_v1.py 启动Web端，程序会自动打开浏览器页面

其他说明：
如果在执行以上文件过程中遇到包缺失的情况，一般只要简单的用pip安装对应包即可
