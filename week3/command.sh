# Create a repository called human-design-rag in ECR
# This is where the Docker Image is stored — Lambda will pull the image from here
aws ecr create-repository --repository-name human-design-rag --region us-east-1

# Log in to ECR so Docker can push images to it
# Replace <YOUR_AWS_ACCOUNT_ID> with your own AWS Account ID (e.g., 123456789012)
aws ecr get-login-password --region us-east-1 | docker login --username AWS --password-stdin <YOUR_AWS_ACCOUNT_ID>.dkr.ecr.us-east-1.amazonaws.com

# Build Docker image
docker build -t human-design-rag .
#  => => naming to docker.io/library/human-design-rag

# Tag the local image with the full ECR address so Docker knows where to push it
docker tag human-design-rag:latest <YOUR_AWS_ACCOUNT_ID>.dkr.ecr.us-east-1.amazonaws.com/human-design-rag:latest

# Push to ECR
docker push <YOUR_AWS_ACCOUNT_ID>.dkr.ecr.us-east-1.amazonaws.com/human-design-rag:latest