from custom_atari_wrapper import atari_wrapper
from explanation import explainer
from explanation import create_lime_image
from explanation import create_saliency_image
from explanation import insertion_for_all
import gym
import keras
import numpy as np
import rise
import datetime
from PIL import Image

if __name__ == '__main__':
    #use a different start to introduce randomnes
    fixed_start = False

    #for reward calculation
    total_reward = 0
    reward_list = []

    np.random.seed(42)

    model = keras.models.load_model('models/SpaceInvadersIngame_5M.h5')

    steps = 1001

    model.summary()
    env = gym.make('SpaceInvadersNoFrameskip-v4')
    env.reset()
    wrapper = atari_wrapper(env)
    wrapper.fire_reset=True
    wrapper.reset(noop_max=1)
    my_explainer = explainer(model=model)
    state_output_path = "space_invaders_explanations/"
    occlusion_auc = []
    lime_auc = []
    greydanus_auc = []
    rise_auc = []
    occlusion_old_confidence_auc = []
    greydanus_auc_noise = []
    scores = []
    if fixed_start :
        wrapper.fixed_reset(1, 0) #used  action 3 and 4
    for _ in range(steps):
        if _ < 4:
            action = env.action_space.sample()
            # to have more controll over the fixed starts
            if fixed_start:
                action=0
        else:
            my_input = np.expand_dims(stacked_frames, axis=0)
            output = model.predict(my_input)  #this output corresponds with the output in baseline if --dueling=False is correctly set for baselines.
            # save model predictions

            action = np.argmax(np.squeeze(output))

            #here you can get the screen frames
            for i in range(len(observations)):
                index = str(_) + '_' + str(i)
                observation = observations[i]

            tmp_scores = []
            if _ % 10 == 0:
                print("Step: " + str(_))
            if _ % 50 == 0:
                plot = True
            else:
                plot = False

            tmp_scores = insertion_for_all(_, my_explainer, stacked_frames, model)
            scores.append(tmp_scores)

        stacked_frames, observations, reward, done, info = wrapper.step(action)
        total_reward += reward
        if done:
            print('total_reward',total_reward)
            reward_list.append(total_reward)
            total_reward = 0
        env.render()

        if _ % 100 == 0 and _ != 0:
            print("Saving progress...")
            np.save(file="figures/backup_space_invaders/pred_" + str(_), arr=scores)
            scores = []

    reward_list.append(total_reward)
    average_reward = np.mean(reward_list)
    print(average_reward)

    print('Time:')
    print(datetime.datetime.now())