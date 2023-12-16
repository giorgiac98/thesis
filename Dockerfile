FROM python:3.10

RUN apt-get install git -y
RUN git clone https://github.com/giorgiac98/thesis.git

WORKDIR /thesis

RUN pip3 install torch --index-url https://download.pytorch.org/whl/cpu && pip3 install -r requirements.txt

WORKDIR /thesis/src
RUN printf "wandb login\nwandb agent \$1\n" > run.sh
RUN ["chmod", "+x", "run.sh"]

ENTRYPOINT ["/bin/sh", "/thesis/src/run.sh"]