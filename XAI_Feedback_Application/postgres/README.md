# Build a custom Postgres image

If you want to customize your Postgres image, you can do so by following 3 steps:

1. Extend the `Dockerfile` in this folder
2. Build a custom image  by adding a job to the your GitLab CI/CD config: `.gitlab-ci.yml`
3. Update your DBMS by calling the `PATCH` endpoint of [Irma Pince API](https://app.roqs.basf.net/irma_pince/api/) with the image you just build in your CI/CD pipeline

## Adding a job to the .gitlab-ci.yml

Add a job that only runs whenever you update the `postgres/Dockerfile`.

```yaml
postgres_build:
  stage: build
  tags:
    - docker
  rules:
    - changes:
      - postgres/Dockerfile
  variables:
    IMAGE_NAME_MARIADB_NUMBERED: ${CI_REGISTRY}/${CI_REGISTRY_NAMESPACE}/postgres:${CI_PIPELINE_ID}
  script:
    - cd postgres
    - docker login -u ${CI_REGISTRY_USER} -p ${CI_REGISTRY_TOKEN} ${CI_REGISTRY}
    - docker build -t $IMAGE_NAME_MARIADB_NUMBERED .
    - docker push $IMAGE_NAME_MARIADB_NUMBERED
```