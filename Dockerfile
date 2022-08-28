FROM ubuntu:20.04 as Build_Env
RUN sed -i "s/security.ubuntu.com/mirrors.aliyun.com/" /etc/apt/sources.list && \
    sed -i "s/archive.ubuntu.com/mirrors.aliyun.com/" /etc/apt/sources.list && \
    sed -i "s/security-cdn.ubuntu.com/mirrors.aliyun.com/" /etc/apt/sources.list

RUN apt-get update 
RUN apt-get install -y vim wget curl g++ gcc make cmake flex bison


