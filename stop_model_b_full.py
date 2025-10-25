import boto3
client = boto3.client('sagemaker', region_name='ap-northeast-1')
client.stop_training_job(TrainingJobName='pytorch-training-2025-10-25-12-39-31-353')
print('Model B full training job stopped')
