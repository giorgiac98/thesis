FROM ubuntu:22.04

RUN apt-get update
RUN apt-get install git -y
RUN git clone https://github.com/giorgiac98/thesis.git

WORKDIR /thesis/src

RUN apt-get install python3 python3-pip -y
RUN pip3 install torch --index-url https://download.pytorch.org/whl/cpu && pip3 install -r requirements.txt

ENV WANDB_API_KEY=ad68d4587494faf541f8b33cf7803ba7a695aaa4

CMD ["wandb", "agent", "giorgiac98/thesis-experiments/v6qpkv4t"]