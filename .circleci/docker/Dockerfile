# this Dockerfile is for torchaudio smoke test, it will be created periodically via CI system
# if you need to do it locally, follow below steps once you have Docker installed
# assuming you're within the directory where this Dockerfile located
#  $ docker build . -t torchaudio/smoketest

# if you want to push to aws ecr, make sure you have the rights to write to ECR, then run
# $ eval $(aws ecr get-login --region us-east-1 --no-include-email)
# $ export MYTAG=localbuild  ## you can choose whatever tag you like
# $ docker tag torchaudio/smoketest 308535385114.dkr.ecr.us-east-1.amazonaws.com/torchaudio/smoke_test:${MYTAG}
# $ docker push  308535385114.dkr.ecr.us-east-1.amazonaws.com/torchaudio/smoke_test:${MYTAG}

FROM ubuntu:latest

RUN apt-get -qq update && apt-get -qq -y install curl bzip2 \
    && curl -sSL https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh -o /tmp/miniconda.sh \
    && bash /tmp/miniconda.sh -bfp /usr/local \
    && rm -rf /tmp/miniconda.sh \
    && conda install -y python=3 \
    && conda update conda \
    && apt-get -qq -y remove curl bzip2 \
    && apt-get -qq -y autoremove \
    && apt-get autoclean \
    && rm -rf /var/lib/apt/lists/* /var/log/dpkg.log \
    && conda clean --all --yes

ENV PATH /opt/conda/bin:$PATH

RUN conda create -y --name python2.7 python=2.7
RUN conda create -y --name python3.5 python=3.5
RUN conda create -y --name python3.6 python=3.6
RUN conda create -y --name python3.7 python=3.7
RUN echo "source /usr/local/etc/profile.d/conda.sh" >> ~/.bashrc
CMD [ "/bin/bash"]
