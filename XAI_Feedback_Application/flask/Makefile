all: build run wait unittest
build: rm
	docker build -t xai_dofc_validation_backend .
run:
	docker run --name xai_dofc_validation_backend -d -p 36555:5000 xai_dofc_validation_backend
wait:
	sleep 5
unittest:
	-docker exec xai_dofc_validation_backend /usr/bin/nosetests -v
	docker logs xai_dofc_validation_backend
stop:
	-docker stop xai_dofc_validation_backend
rm: stop
	-docker rm xai_dofc_validation_backend
rmi: stop rm
	-docker rmi xai_dofc_validation_backend
