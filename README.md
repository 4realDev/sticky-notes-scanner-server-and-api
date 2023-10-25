# build

```
docker build --secret id=gitlab-pat,src=./.gitlab_pat --tag toolbox-scanning-api .
```

# run (production)
```
docker run -v$(pwd)/.gcloud_serviceaccount.json:/.gcloud_serviceaccount.json -e GOOGLE_APPLICATION_CREDENTIALS=/.gcloud_serviceaccount.json --rm -p8080:8080 -it toolbox-api-scanning
```
