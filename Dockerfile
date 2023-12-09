FROM ubuntu:22.04

RUN apt-get update
RUN apt-get install git -y
RUN git clone https://github.com/giorgiac98/thesis.git

WORKDIR /thesis

RUN apt-get install python3 python3-pip -y
RUN pip3 install torch --index-url https://download.pytorch.org/whl/cpu && pip3 install -r requirements.txt

WORKDIR /thesis/src
RUN printf "wandb login\nwandb agent \$1\n" > run.sh
RUN ["chmod", "+x", "run.sh"]

ENTRYPOINT ["/bin/sh", "/thesis/src/run.sh"]