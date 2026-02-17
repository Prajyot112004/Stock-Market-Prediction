pipeline {
    agent any
    stages {
        stage('Build & Setup') {
            steps {
                echo 'Preparing environment...'
                checkout scm
            }
        }
        stage('Deploy Streamlit App') {
            steps {
                echo 'Deploying to Docker...'
                bat '''
                    docker build -t stock-app .
                    docker stop stock-container || true
                    docker rm stock-container || true
                    docker run -d --name stock-container -p 8501:8501 stock-app
                '''
            }
        }
        stage('Selenium Testing') {
            steps {
                echo 'Waiting for app...'
                bat 'timeout /t 15' 
                echo 'Running Selenium Regression Test...'
                bat 'python -u test_login.py'
            }
        }
    }
}
