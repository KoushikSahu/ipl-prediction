all: build run

build: submission/submission/*.pkl
	cp bin/objs/*.pkl submission/submission/
	cp bin/models/*.pkl submission/submission/

run:
	cd submission/submission && python3 main.py

clean:
	rm -rf submission/submission/*.pkl
