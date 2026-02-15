pipeline {
    agent any

    stages {
        stage('Build & Setup') {
            steps {
                echo 'Preparing environment for Stock Prediction...'
                // This pulls your code from GitHub
                checkout scm
            }
        }
        stage('Model Training') {
            steps {
                echo 'Training model on historical data...'
                // Note: In a real MLOps pipeline, you would run 'python train.py' here
            }
        }
        stage('Deploy Streamlit App') {
            steps {
                echo 'Deploying to Docker container...'
                bat '''
                    docker build -t stock-prediction-app .
                    docker stop stock-container || true
                    docker rm stock-container || true
                    docker run -d --name stock-container -p 8501:8501 stock-prediction-app
                '''
            }
        }
    }
}