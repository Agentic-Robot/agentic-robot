from ds import decompose_task_with_llm 

my_complex_task = "put_the_black_bowl_in_the_bottom_drawer_of_the_cabinet_and_close_it"

planned_subtasks = decompose_task_with_llm(my_complex_task)

        # "LIVING_ROOM_SCENE2_put_both_the_alphabet_soup_and_the_tomato_sauce_in_the_basket",
        # "LIVING_ROOM_SCENE2_put_both_the_cream_cheese_box_and_the_butter_in_the_basket",
        # "KITCHEN_SCENE3_turn_on_the_stove_and_put_the_moka_pot_on_it",
        # "KITCHEN_SCENE4_put_the_black_bowl_in_the_bottom_drawer_of_the_cabinet_and_close_it", #####
        # "LIVING_ROOM_SCENE5_put_the_white_mug_on_the_left_plate_and_put_the_yellow_and_white_mug_on_the_right_plate",
        # "STUDY_SCENE1_pick_up_the_book_and_place_it_in_the_back_compartment_of_the_caddy",
        # "LIVING_ROOM_SCENE6_put_the_white_mug_on_the_plate_and_put_the_chocolate_pudding_to_the_right_of_the_plate",
        # "LIVING_ROOM_SCENE1_put_both_the_alphabet_soup_and_the_cream_cheese_box_in_the_basket",
        # "KITCHEN_SCENE8_put_both_moka_pots_on_the_stove",
        # "KITCHEN_SCENE6_put_the_yellow_and_white_mug_in_the_microwave_and_close_it", #######
# if planned_subtasks:
#     print("Successfully decomposed task:")
#     for step, subtask in enumerate(planned_subtasks):
#         print(f"  {step + 1}: {subtask}")
#   
#         # e.g., execute_vla_command(subtask)
# else:
#     print(f"Could not decompose task: {my_complex_task}")
print(planned_subtasks)