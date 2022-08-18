FROM tione-wxdsj.tencentcloudcr.com/base/pytorch:py38-torch1.9.0-cu111-1.0.0
# FROM tione-wxdsj.tencentcloudcr.com/base/pytorch:py38-torch1.9.0-cu111-trt8.2.5


# WORKDIR /opt/ml/wxcode

COPY ./requirements.txt ./
RUN pip install -r requirements.txt -i https://mirrors.cloud.tencent.com/pypi/simple

COPY ./src ./src

COPY ./install.sh ./
RUN sh ./install.sh

COPY ./opensource_models ./opensource_models

COPY ./*.sh ./