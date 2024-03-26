FROM ubuntu:20.04

# Install Java and other dependencies
RUN apt-get update -y \
    && apt-get install openjdk-8-jdk -y \
    && apt-get install python3-pip -y \
    && apt-get install -y wget

RUN adduser user
RUN wget https://dlcdn.apache.org/spark/spark-3.3.1/spark-3.3.1-bin-hadoop3.tgz
RUN tar xvf spark-3.3.1-bin-hadoop3.tgz -C /opt
RUN chown -R user:user /opt/spark-3.3.1-bin-hadoop3
RUN ln -fs spark-3.3.1-bin-hadoop3 /opt/spark
RUN echo -e "export SPARK_HOME=/opt/spark\nPATH=$PATH:$SPARK_HOME/bin\nexport PATH" >> ~/.bash_profile
RUN . ~/.bash_profile
COPY --chown=User:user . .
RUN pip3 install -r requirements.txt
ENTRYPOINT ["runuser", "-u", "user", "--", "python3"]
CMD ["testing.py"]