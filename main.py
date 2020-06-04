import sc2


if __name__ == "__main__":
    image = sc2.utils.image_load("jinge")
    gradient = sc2.energy.entropy_energy(image)
    sc2.utils.image_show(gradient, freeze=True)