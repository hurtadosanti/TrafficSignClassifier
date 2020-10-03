FROM tensorflow/tensorflow:latest-gpu
RUN pip3 install numpy pandas jupyter matplotlib scikit-learn

CMD jupyter notebook --ip 0.0.0.0 --allow-root