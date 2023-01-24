import get_descriptions
import streamlit as st
import torch
import utils

st.set_page_config(page_title = 'PIBIC GANs', layout = 'wide', page_icon = '🎨')
st.markdown("<div id='inicio' style='visibility: hidden'></div>", unsafe_allow_html = True)
st.markdown('<style> #MainMenu {visibility: hidden;} footer {visibility: hidden;} </style>', 
            unsafe_allow_html = True)

padding = 1
st.markdown(f"""<style>
                .reportview-container .main .block-container{{
                    padding-top: {padding}rem;
                    padding-right: {padding}rem;
                    padding-left: {padding}rem;
                    padding-bottom: {padding}rem;}}
                </style>""", unsafe_allow_html = True)

st.header(body = 'Geração de Imagens Artificiais para Aumento de Dados Utilizando Redes Adversárias '
                 'Generativas (GANs)')
st.markdown(body = '**Objetivo da Pesquisa**: O objetivo desse projeto é o desenvolvimento de um sistema '
                   'que permita o aumento de dados em bases de imagens a partir da geração de amostras '
                   'sintéticas utilizando GANs e outros algoritmos de Sistemas Generativos. Também é '
                   'esperado que o sistema gere imagens artificiais com uma boa representação visual, '
                   'diversidade e fidelidade com o grau de detalhamento presente nas imagens originais '
                   'das bases de dados testadas.')

data_box = st.selectbox(label = 'Selecione a Base de Dados:', options = ['MNIST', 'CELEBA (Indisponível)', 
                        'CIFAR - 10 (Indisponível)', 'FASHION - MNIST (Indisponível)'], index = 0, 
                        help = 'Selecione a base de dados para testar a capacidade dos algoritmos '
                        'generativos disponíveis para teste.')
algorithm_box = st.selectbox(label = 'Selecione o Sistema Generativo:', 
                             options = ['Simple GAN (Indisponível)', 'DCGAN', 'SNGAN (Indisponível)', 
                             'WGAN - GP (Indisponível)', 'SNGAN + WGAN - GP (Indisponível)'], index = 1, 
                             help = 'Selecione o sistema generativo para visualizar as imagens '
                             'artificiais geradas de acordo com a base de dados selecionada.')

st.markdown(body = f'{data_box} e {algorithm_box}')

st.markdown(body = f'### Base de Dados: {data_box} | Sistema Generativo: {algorithm_box}')
st.markdown(body = '***')

if data_box == 'MNIST' and algorithm_box == 'DCGAN':

    z_dim = 64
    mnist_shape = (1, 28, 28)
    n_classes = 10
    device = 'cpu'

    generator_input_dim, discriminator_im_chan = utils.get_input_dimensions(z_dim, mnist_shape, n_classes)
    gen = utils.Generator(input_dim = generator_input_dim).to(device)
    gen.load_state_dict(torch.load('models/gen_conditional_dcgan.pth', 
                        map_location = torch.device(device)))
    gen.eval()

    col1, col2, col3 = st.columns(spec = [5, 1, 1])

    with col1:
        start_plot_number = st.slider(label = 'Selecione o primeiro dígito para interpolação:', 
                                      min_value = 0, max_value = 9, value = 3)
        end_plot_number = st.slider(label = 'Selecione o segundo dígito para interpolação:',
                                    min_value = 0, max_value = 9, value = 9)
        n_interpolation = st.slider(label = 'Selecione a quantidade de imagens geradas na interpolação:',
                                    min_value = 10, max_value = 200, value = 100)
    
    with col2:
        images = utils.interpolate_class(first_number = start_plot_number, 
                                         second_number = end_plot_number, z_dim = z_dim, 
                                         n_classes = n_classes, device = device, gen = gen, 
                                         n_interpolation = n_interpolation, show_tensor = False)
        utils.create_gif(images_pytorch_tensor = images)
        
        st.image(image = 'images/tensor_images.gif', width = 150)

        images = utils.interpolate_class(first_number = start_plot_number, 
                                         second_number = end_plot_number, z_dim = z_dim, 
                                         n_classes = n_classes, device = device, gen = gen, 
                                         n_interpolation = n_interpolation, show_tensor = False)
        utils.create_gif(images_pytorch_tensor = images)
        
        st.image(image = 'images/tensor_images.gif', width = 150)
    
    with col3:
        images = utils.interpolate_class(first_number = start_plot_number, 
                                         second_number = end_plot_number, z_dim = z_dim, 
                                         n_classes = n_classes, device = device, gen = gen, 
                                         n_interpolation = n_interpolation, show_tensor = False)
        utils.create_gif(images_pytorch_tensor = images)

        st.image(image = 'images/tensor_images.gif', width = 150)

        images = utils.interpolate_class(first_number = start_plot_number, 
                                         second_number = end_plot_number, z_dim = z_dim, 
                                         n_classes = n_classes, device = device, gen = gen, 
                                         n_interpolation = n_interpolation, show_tensor = False)
        utils.create_gif(images_pytorch_tensor = images)
        
        st.image(image = 'images/tensor_images.gif', width = 150)
    
    st.markdown(body = '***')

    col1, col2, col3 = st.columns(spec = [2, 1, 1])

    with col1:
        n_interpolation = st.slider(label = 'Dígite a quantidade de imagens geradas na interpolação:', 
                                    min_value = 6, max_value = 30, value = 9, step = 3)
        n_noise = st.slider(label = 'Selecione o dígito a ser gerado:', min_value = 0, max_value = 9,
                            value = 9)
        interpolation_label = utils.get_one_hot_labels(torch.Tensor([n_noise]).long(), 
                                                       n_classes).repeat(n_interpolation, 1).float()

    with col2:
        utils.interpolate_noise_gif(n_noise = n_noise, z_dim = z_dim, n_interpolation = n_interpolation, 
                                    interpolation_label = interpolation_label, gen = gen, device = device)
        st.image(image = 'images/interpolate-noise.gif', width = 300) 

    with col3:
        utils.interpolate_noise_gif(n_noise = n_noise, z_dim = z_dim, n_interpolation = n_interpolation, 
                                    interpolation_label = interpolation_label, gen = gen, device = device)
        st.image(image = 'images/interpolate-noise.gif', width = 300) 
    
    st.markdown(body = '#### Detalhes sobre o Sistema Generativo:')
    st.markdown(body = get_descriptions.dcgan_description())
    st.image(image = 'data/dcgan-gen.png', caption = 'Desenho arquitetural do gerador da DCGAN '
             'Radford et al (2016). Disponível em: https://arxiv.org/pdf/1511.06434v1.pdf.')
    st.markdown(body = '***')
    st.markdown(body = '#### Detalhes sobre a Base de Dados:')
    st.markdown(body = get_descriptions.mnist_description())
else:
    st.warning(body = 'Desculpe, mas essa combinação de base de dados e sistema generativo está indisponível '
               'no momento.')
