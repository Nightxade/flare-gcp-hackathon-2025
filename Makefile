.PHONY: build run clean

all: run

build:
	docker build -t flare-ai-rag .

run: build
	docker run -p 80:80 -it --env-file .env flare-ai-rag