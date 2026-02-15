pipeline {
    agent any
    stages {
        stage('Build & Setup') {
            steps {
                echo 'Preparing environment for Stock Prediction...'
            }
        }
        stage('Model Training') {
            steps {
                echo 'Training model on historical data...'
            }
        }
        stage('Results') {
            steps {
                echo 'Displaying prediction accuracy...'
            }
        }
	stages {
        stage('Checkout') {
            steps {
                // This pulls your code from GitHub
                checkout scm
            }
        }
        stage('Deploy Streamlit App') {
            steps {
                // 'bat' is for Windows. If you were on Linux, you'd use 'sh'
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
}