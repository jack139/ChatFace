# TEST


## 环境

- 需要 torch>=2.1

- 单独安装 TTS

```bash
sudo pip3.9 install --no-deps TTS==0.22.0
```



## pydantic-2.6.3 使用 v1 的方法：

例如 spacy 中，加入```v1```：

```python
# /usr/local/lib/python3.9/site-packages/spacy/schemas.py
from pydantic.v1 import BaseModel, Field, ValidationError, validator, create_model
from pydantic.v1 import StrictStr, StrictInt, StrictFloat, StrictBool
from pydantic.v1.main import ModelMetaclass
```



## 测试

```bash
python3.9 test.py
```