import matplotlib.pyplot as plt

reward = []
with open('score.txt','r') as f:
	for line in f.readlines():
		left, right = line.strip().split(':')
		reward.append(int(left)-int(right))

plt.plot(list(range(1, len(reward)+1)), reward)
plt.xlabel('Episode')
plt.ylabel('Reward')
plt.title('Learning Curve')
plt.savefig('reward.png')