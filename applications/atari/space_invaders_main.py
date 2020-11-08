from custom_atari_wrapper import atari_wrapper
from explanation import explainer
from explanation import create_lime_image
from explanation import create_saliency_image
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
            saliency_map = my_explainer.generate_occlusion_explanation(input=np.squeeze(stacked_frames), use_softmax=True)
            # create_saliency_image(saliency_map=saliency_map, image=observations[3],
            #                                        output_path=state_output_path + "occlusion_explanation_" + "frame_" + str(_), cmap="viridis")
            insertion = rise.CausalMetric(model=model, mode='ins', step=np.squeeze(stacked_frames).shape[0],
                                                      substrate_fn=rise.custom_blur)
            score = insertion.single_run(img_tensor=np.squeeze(stacked_frames), explanation=saliency_map, name="frame_" + str(_), approach="occl", plot=plot)
            tmp_scores.append(score)
            occlusion_auc.append(rise.auc(score))

            saliency_map = my_explainer.generate_occlusion_explanation(input=np.squeeze(stacked_frames),
                                                                       use_softmax=True, use_old_confidence=True)
            score = insertion.single_run(img_tensor=np.squeeze(stacked_frames), explanation=saliency_map,
                                         name="frame_" + str(_), approach="occl_old_pred", plot=plot)
            tmp_scores.append(score)
            occlusion_old_confidence_auc.append(rise.auc(score))

            saliency_map = my_explainer.generate_greydanus_explanation(input=np.squeeze(stacked_frames), blur=False)
            # create_saliency_image(saliency_map=saliency_map, image=observations[3],
            #                                       output_path=state_output_path + "greydanus_explanation_" + "frame_" + str(_))
            score = insertion.single_run(img_tensor=np.squeeze(stacked_frames), explanation=saliency_map, name="frame_" + str(_),
                                                     approach="noise", plot=plot)
            tmp_scores.append(score)
            greydanus_auc.append(rise.auc(score))

            saliency_map = my_explainer.generate_greydanus_explanation(input=np.squeeze(stacked_frames), blur=True)
            score = insertion.single_run(img_tensor=np.squeeze(stacked_frames), explanation=saliency_map,
                                         name="frame_" + str(_),
                                         approach="noise_blur", plot=plot)
            tmp_scores.append(score)
            greydanus_auc_noise.append(rise.auc(score))

            explanation, mask, ranked_mask = my_explainer.generate_lime_explanation(rgb_image=False,
                                                                                            input=np.squeeze(stacked_frames),
                                                                                            hide_img=False, positive_only=False)
            # create_lime_image(mask=mask, frames=observations,
            #                               output_path=state_output_path + "lime_explanation_" + "frame_" + str(_), shape=(160, 210))
            score = insertion.single_run(img_tensor=np.squeeze(stacked_frames), explanation=ranked_mask, name="frame_" + str(_), approach="lime", plot=plot)
            tmp_scores.append(score)
            lime_auc.append(rise.auc(score))

            saliency_map = my_explainer.generate_rise_prediction(input=np.squeeze(stacked_frames))
            # create_saliency_image(saliency_map=saliency_map, image=observations[3],
            #                                   output_path=state_output_path + "rise_explanation_" + "frame_" + str(_))
            score = insertion.single_run(img_tensor=np.squeeze(stacked_frames), explanation=saliency_map, name="frame_" + str(_),
                                                     approach="rise", plot=plot)
            tmp_scores.append(score)
            rise_auc.append(rise.auc(score))
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

    print("Occlusion average AUC: ")
    print(sum(occlusion_auc) / len(occlusion_auc))
    print("LIME average AUC:")
    print(sum(lime_auc) / len(lime_auc))
    print("Greydanus average AUC:")
    print(sum(greydanus_auc) / len(greydanus_auc))
    print("RISE average AUC:")
    print(sum(rise_auc) / len(rise_auc))