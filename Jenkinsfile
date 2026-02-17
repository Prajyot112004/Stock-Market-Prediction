pipeline {
    agent any

    stages {
        stage('Build & Setup') {
            steps {
                echo 'Preparing environment for Stock Prediction...'
                checkout scm
            }
        }
        stage('Model Training') {
            steps {
                echo 'Training model on historical data...'
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
        stage('Selenium Testing') {
            steps {
                echo 'Waiting for app to initialize...'
                // Wait for 15 seconds to ensure Streamlit is fully loaded
                bat 'timeout /t 15 /nobreak' 
                
                echo 'Running UI Tests...'
                // Use -u to make sure you see the "Passed" message in logs
                bat 'python -u test_login.py'
            }
        }
    }
}