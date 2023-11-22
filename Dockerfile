FROM ubuntu:20.04 as Build_Env
#RUN sed -i "s/security.ubuntu.com/mirrors.aliyun.com/" /etc/apt/sources.list && \
#    sed -i "s/archive.ubuntu.com/mirrors.aliyun.com/" /etc/apt/sources.list && \
#    sed -i "s/security-cdn.ubuntu.com/mirrors.aliyun.com/" /etc/apt/sources.list


RUN apt-get update 
RUN DEBIAN_FRONTEND=noninteractive apt-get install -y \
    vim wget curl g++ gcc make cmake flex bison libeigen3-dev \
    libgtest-dev libceres-dev git apt-transport-https gnupg


RUN curl -fsSL https://bazel.build/bazel-release.pub.gpg | gpg --dearmor >bazel-archive-keyring.gpg
RUN mv bazel-archive-keyring.gpg /usr/share/keyrings
RUN echo "deb [arch=amd64 signed-by=/usr/share/keyrings/bazel-archive-keyring.gpg] https://storage.googleapis.com/bazel-apt stable jdk1.8" | tee /etc/apt/sources.list.d/bazel.list
RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y bazel

RUN git clone https://github.com/junlinp/Optimization.git

#RUN mkdir -p Optimization/build && cd Optimization/build && cmake .. && make -j
RUN cd Optimization && bazel build ...

