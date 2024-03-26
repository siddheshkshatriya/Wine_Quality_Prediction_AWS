# Wine-quality-prediction-aws
CS-643- Programming Assignment-2

## Parallel Training Implementation: - ##
---------------

- Cluster Creation: - We will create 1 cluster with 5 nodes where 1 node will act as master node and others will be slave nodes.

- File uploading on S3 bucket: - After cluster creation S3 bucket will be auto generated. Here we will upload Training.py file and the dataset. 

- Now we will pull the stored files from S3 bucket to the master node by connecting to the master node (Using following commands).

		aws s3 cp s3://aws-logs-877244108283-us-east-1/elasticmapreduce/j-38II9TYTEUBU5/TrainingDataset.csv ./
    	aws s3 cp s3://aws-logs-877244108283-us-east-1/elasticmapreduce/j-38II9TYTEUBU5/ValidationDataset.csv ./
    	aws s3 cp s3://aws-logs-877244108283-us-east-1/elasticmapreduce/j-38II9TYTEUBU5/Training.py ./
    
- Now we will make these files available to other slave nodes(Using following commands).
     hadoop fs -put TrainingDataset.csv
		 hadoop fs -put ValidationDataset.csv
     
- Type “ls” and hit enter you will find all the files stored in it.

- Install all the required libraries by storing it in “requirements.txt”(We’ve already uploaded it in S3 bucket) and running the following command.
			pip install -r requirements.txt
	
- Scala Installation on master node: -
			wget https://downloads.lightbend.com/scala/2.12.4/scala-2.12.4.rpm
			sudo yum install scala-2.12.4.rpm

- Spark Installation: -
			wget https://dlcdn.apache.org/spark/spark-3.3.1/spark-3.3.1-bin-hadoop3.tgz
      sudo tar xvf spark-3.3.1-bin-hadoop3.tgz -C /opt
      sudo chown -R ec2-user:ec2-user /opt/spark-3.3.1-bin-hadoop3
      sudo ln -fs spark-3.3.1-bin-hadoop3 /opt/spark
	 

- Let’s run the Training.py using: - 
- spark-submit Training.py


## Predicting the wine quality on single machine: - ##
---------------

- Create new ec2 instance and connect to it.
- Install python, Java, Spark, Scala, and all the required dependencies.
- The trained model will get store in S3 bucket lets copy it from S3 to ec2 instance: - 

      aws s3 cp s3://aws-logs-766621730595-us-east-1/elasticmapreduce/j-1CPN0XQGUGAEC/trainingmodel.model ./ --recursive


- Now Unzip the model and move the contents to the new folder which we will be creating now: - 	

    tar -xzvf model.tar.gz
    mkdir model
    mv data<downloaded file> model<model folder>
    mv metadata<downloaded file> model<model folder>

- Update the path Environment: -
	
      vim ~/.bash_profile
      copy following lines into file and then save it
      export SPARK_HOME=/opt/spark
      PATH=$PATH:$SPARK_HOME/bin
      export PATH

- Save the file and run the following command: - 

      source  ~/.bash_profile

- Run the Testing file: - 
	
        spark-submit test.py


  
## Prediction using Docker Container:- ##
---------------
	
- The container runs the validationData.csv and prints Test Error Link to docker hub image

- Launch your ec2-instance and then step-up docker using the above steps.
- Place all the files from github into your instance.
- Pull the image from repositroy: 
	
		docker pull tejashk/win_quality_prediction

- Run the image using : 
	
		docker run tejashk/win_quality_prediction driver test.py ValidationDataset.csv model


  

