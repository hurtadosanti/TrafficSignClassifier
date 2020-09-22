FROM tensorflow/tensorflow:1.15.0-gpu-py3
RUN apt-get update && apt-get -yq upgrade && apt install -y wget
RUN pip3 install scikit-learn jupyter pandas matplotlib
RUN pip3 install --upgrade pip
#docker run -it --runtime=nvidia -p 8888:8888 -v $PWD/:/home/shurtado local/tensorflow-gpu:0.1
#jupyter notebook --port=8888 --no-browser --ip=0.0.0.0 --allow-root