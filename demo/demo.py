from image_captioning import image_captioning, Vocabulary

if __name__ == '__main__':
    caption_set = image_captioning('./surf.jpg')

    print(caption_set)
    for caption in caption_set:
        print(caption.strip('<sos> ').strip(' <eos>'))
