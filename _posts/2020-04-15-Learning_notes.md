# Learning Notes

- [Docker](##docker)
- [Github](##github)
- [markdown](##markdown)

This blog contains some of the stuffs I have learnt and learning. I want to put them in one place for my personal references and call backs. This blog mostly contains the very basics item and stairs to the complex item. I will be putting the basic things with a probable next steps with basics. For data-science related development please follow my other github repositories.  

I am also planning to expand the notes by putting some of the common stuffs that bugs me. Lets kill the bug!


## Cleaning GPU linux

In the terminal type "nvtop"

Find and note the activity PID with most GPU usage. Lets assume the number is 85954 (a random one).

Exit the nvtop place and type the following, of course replace the 85954 with the noted number.

```
sudo kill PID -85954

```

If it doesn't work, I am afraid it might require the reboot :( .

### Bluetooth Headphone connection Ubuntu
Well I did many things
The following worked finally. not sure if it was sole responsible of sum of many previous things. However,
https://kumarvinay.com/how-to-enable-bluetooth-headset-microphone-support-in-ubuntu-20-04/

After following those, I still require to run the following command everytime after I reconnect my bluetooth device. 

```
systemctl --user --now enable pipewire-media-session.serviceon.
```

## Docker

The most important thing regarding the docker is the concept of images and container. Images stay in locally, we open container from images each time. There are vice versa too. The image can be think of the class and container are class instances. Container starts and gets deleted unless there are volume assignment. The docker is used to run virtual machine inside os or virtual os. There are tons of tutorial available. I am going to make my own collection based on other.
[tutorial-1](https://jonnylangefeld.github.io/learning/Docker/How%2Bto%2BDocker.html)

The first thing we want to run after installation is
```
docker run hello-world
```
This will bring an images and container. Here is the fact, we need to remove the container after the work but image will be there for further use, just like a defined class. To see what containers are active-
```
docker ps -a
```
Now we can create a container from a image using custom name using docker run just like before but using some more parameters.
```
docker run --name random_hello hello-world
```
To remove the docker container we created earlier. The way to get out of container is use of exit command.
```
docker rm random_hello
```
We can also keep the docker container but stop it from executing.
```
docker kill random_hello
```
We can start docker and pause docker containers
```
docker start hello_random
docker pause hello_random
```
But unpause before reconnect again
```
docker unpause hello_random
docker exec it hello_random bash # if it has bash anyway
```
The exec command execute command in a running container. [for more](https://docs.docker.com/engine/reference/commandline/container_attach/) and [alternative](https://docs.docker.com/engine/reference/commandline/docker/)

Now the interesting part; getting ubuntu inside ubuntu for me
```
docker run -it --name my-linux-container ubuntu bash
```
This will create a container from ubuntu images and directly go to bash file. Great. But anything you create insider will be deleted with the container removal. We can see now what containers are there and how many container is running. We can create another container. Be careful of the name collision. As we already have created my-linux-container earlier.
```
docker run -it --name my-linux-container-1 ubuntu bash
```
We can delete each container by name/Id or all by the following.
```
docker rm $(docker ps -a -f status=exited -q)
```
It will remove the exited container only. Now we want the docker to look at some local files to work with.
```
docker run -it --name my-linux-container --rm -v path_to_my-data ubuntu bash
```
It will provide me the original folder under the ubuntu environment. Some important removal [notes](https://www.digitalocean.com/community/tutorials/how-to-remove-docker-images-containers-and-volumes)
As we have seen three things: image, volume and containers. There is a master command
```
docker system prune # or prune -a
```
Or we can go each by each
```
# image removal
docker images -a
docker rmi {image_name}
docker images prune
docker rmi $(docker images -a -q)

# Container removal
docker ps -a
docker rm ID_or_Name

# Removing volume
docker volume ls
docker volume rm volume_name
docker volume prune
docker rm -v container_name # remove volume from container
```
Another interesting way to remove container by stating rm after run command.
```
docker run --rm image_name
docker ps -a -f status=exited
docker rm $(docker ps -a -f status=exited -q)
```
Now more interesting thing Dockerfile. A way to define personalized images. For that we need a file saved a Dockerfile and inside the file we can write
```
FROM ubuntu
CMD echo "Anything else"
```
This command tells to build a image using ubuntu and perform a echo command in bash. We can build another image on that by running the commands on same files
```
docker build -t my-ubuntu-image .
```
Now we can check the images. Its name will be my-ubuntu-image. So basically Dockerfile gives us an image and we can use it in our own ways to make container discussed earlier.
```
docker images
```
We can run the images
```
docker run my-ubuntu-image
```
We can do more in Dockerfile
```
FROM ubuntu
RUN apt-get upgrade && apt-get update && apt-get install -y python3
```
Now the same routine docker build and then run the image instances in a container. Last not the least is the Docker compose. Align multiple image together.
```
Ask Question Before you Learn. Search the right question.
```
## Github

[Uncommit](https://stackoverflow.com/questions/2845731/how-to-uncommit-my-last-commit-in-git) last commit (Will keep the modified changes in your working tree.)

```
git reset --soft HEAD^
```

However, care about the following, as it WILL THROW AWAY THE CHANGES YOU MADE !!!
```
git reset --hard HEAD^
```

## MarkDown

## Python Super

The first thing in my comes is class on top of class and class inheritance. [link](https://realpython.com/python-super/)

As example talks best in coding
```
%python
class Rectangle:
    def __init__(self, length, width):
        self.length = length
        self.width = width

    def area(self):
        return self.length * self.width

    def perimeter(self):
        return 2 * self.length + 2 * self.width

class Square(Rectangle):
    def __init__(self, length):
        super(Square, self).__init__(length, length) % same as super().__init__(...)
```

Here, Square class is an instance of Rectangle class by making the length equals [Square(Rectangle) define instances]. The super makes the square a new super class for future references. **The super(...) allows to call the method of superclass (rectangle) in the subclass (square).**

We can inherit multiple classes and their methods.   

[Fun with super](https://www.programiz.com/python-programming/methods/built-in/super)


## Ubuntu Problems and solution

Well recently I encountered a problem of internal microphone. It wasn't working. Well, I was restarting and trying bunch of thing, still not working. Then

- Totally turn PC off for some time and boom, started manually and worked again :)

### Ubuntu Brightness issues


http://sandipbgt.com/2015/10/01/control-screen-brightness-from-commandline-in-ubuntu/


### Ubuntu Sending Email
https://www.cier.tech/blog/tech-tutorials-4/post/how-to-install-and-configure-sendmail-on-ubuntu-35


https://gist.github.com/adamstac/7462202

write the text.txt file and ka boom

```
sendmail -t sender@domain.com -t <text.txt
```

# Conda issues

### The environment is inconsistent, please check the package plan carefully The following packages are causing the inconsistency:

Try the following in the terminal:

```
conda install anaconda

conda install conda

conda update conda
```

[link](https://stackoverflow.com/questions/55527354/the-environment-is-inconsistent-please-check-the-package-plan-carefully)

If the problem persists, use the following

```
conda remove conda
conda remove anaconda
conda install conda
conda install anaconda
```

If the problem still persists, remove the problematic package manually by

```
conda remove "PACKAGE_name"
```
### ModuleNotFoundError: No module named 'torchvision.ops'

Remove the pytorch from the environment and reinstall everything. (Cuda 10.2 version worked for me).

Install the **right version** of the pytorch, torchvision. (try both conda and pip for the right version)

### plt.imshow taking forever/ not working

Use plt.show() at the end of plt.imshow()


# Initramfs Ubuntu reboot

Prospective solution to the
https://proposedsolution.com/solutions/ubuntu-booting-to-initramfs-prompt/

```
type "exit"
```

Find the error in particular sata port
/dev/sdaX: Unexpected ....

/dev/sdax - X for the number

then type
```
(initramfs)fsck /dev/sda1
```

### jupyter notebook reload python file

Refresh the jupyer notebook and we should get the updated py files.

```
import importlib
importlib.reload(python_file_name_without_dot_py)
```


[link](https://www.pauldesalvo.com/how-to-refresh-an-imported-python-file-in-a-jupyter-notebook/)

## Tensorflow GPU memory hack
use the following instead of just running the network to free gpu memory after kernel restart.
```
with tf.device('gpu:0'): #very important
    trainNet(neural_net)
```
