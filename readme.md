# This repository is implementaion of diploma thesis Object Manipulation with Large language Models in Real Robots


## Setting up the server site
As server I used karolina supercomputer but anythig else with suficient computing power is sufficient.

I use port fowarding for interference between the robot and LLMs

Foward to karolina 
```bash
ssh -i SSH_KEY -L 5000:localhost:5000 USERNAME@karolina.it4i.cz
```

Connect to specific node
```bash
ssh -L 5000:localhost:5000 NODE_NAME 
```

The server side has to have the following structure

```
├────KMR-object-manipulation-with-LLMs (this project)
├────megapose6d
```

[megapose6d](https://github.com/megapose6d/megapose6d)
