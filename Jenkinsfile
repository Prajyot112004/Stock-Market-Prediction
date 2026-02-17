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
                echo 'Waiting for app to initialize...'
                // This is the Windows way to wait without crashing Jenkins
                bat 'ping 127.0.0.1 -n 20 > nul' 
                
                echo 'Running Selenium Regression Test...'
                // Use the full path or just 'python' if it's in your PATH
                bat 'python -u test_login.py'
            }
        }
    }
}
