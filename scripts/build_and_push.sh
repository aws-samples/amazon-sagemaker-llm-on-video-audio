
#!/usr/bin/env bash
# This script shows how to build the Docker image and push it to ECR to be ready for use
# by SageMaker.
# The argument to this script are the path to the Dockerfile, the image name and tag and the aws-region
# in which the container is to be created. This will be used as the image on the local
# machine and combined with the account and region to form the repository name for ECR.

# override the built-in echo so that we can have a nice timestamped trace
echo () {
    builtin echo "$(date +'[%m-%d %H:%M:%S]'):" "$@"
}

if [ "$#" -eq 4 ]; then
    dlc_account_id=$(aws sts get-caller-identity | jq .Account)
    path_to_dockerfile=$1
    image=$2
    tag=$3
    region=$4
    
else
    echo "missing mandatory command line arguments, see usage..."
    echo "usage: $0 </path/to/Dockerfile> $1 <image-repo> $2 <image-tag> $3 <aws-region>"
    exit 1
fi

# Get the account number associated with the current IAM credentials
account=$(aws sts get-caller-identity --query Account --output text)

if [ $? -ne 0 ]
then
    exit 255
fi


fullname="${account}.dkr.ecr.${region}.amazonaws.com/${image}:${tag}"
echo the full image name would be ${fullname}

# If the repository doesn't exist in ECR, create it.
aws ecr describe-repositories --region ${region} --repository-names "${image}" > /dev/null 2>&1
if [ $? -ne 0 ]; then
    echo "creating ECR repository : ${fullname} "
    aws ecr create-repository --region ${region} --repository-name "${image}" > /dev/null
else
    echo "${image} repo already exists in ECR"
fi

# move to path of dockerfile
cd ${path_to_dockerfile}

# get credentials to login to ECR and, build and tag the image
# note the use of DOCKER_BUILDKIT=1, this is needed for some mount instructions in the Dockerfile
echo "going to start a docker build, image=${image}, using Dockerfile=${path_to_dockerfile}"
aws ecr get-login-password --region ${region} \
| docker login --username AWS --password-stdin ${account}.dkr.ecr.${region}.amazonaws.com
DOCKER_BUILDKIT=1 docker build . -t ${image}  --build-arg dlc_account_id=${dlc_account_id} --build-arg region=${region}
docker tag ${image} ${fullname}
echo ${image} created

# push the image to ECR
cmd="aws ecr get-login-password --region ${region} | docker login --username AWS --password-stdin ${account}.dkr.ecr.${region}.amazonaws.com"
echo going to run \"${cmd}\" to login to ECR
${cmd}

cmd="docker push ${fullname}"
echo going to run \"${cmd}\" to push image to ecr
${cmd}
if [ $? -eq 0 ]; then
    echo "Amazon ECR URI: ${fullname}"
else
    echo "Error: Image ${fullname} build and push failed"
    exit 1
fi

echo "all done"
