# **EFFICIENT-CAPSNET: CAPSULE NETWORK WITH SELF-ATTENTION ROUTING**  
  
Tensorflow 2.x (with Keras API) Implementation of the Efficient-CapsNet (Mazzia et al., 2021)

# **Training and Testing (MNIST)**  
  
Use
```bash
python main.py --checkpoint_dir=checkpoint --num_epochs=15
```  
- checkpoint: path to save trained weights of the model.
- num_epochs: Number of epochs. You can resume any previous training.
  
You can also simplify  
```bash
python main.py --flagfile=./flags.txt
```
  
# **References**
- Efficient-CapsNet: Capsule Network with Self-Attention Routing ([arXiv][efficient_capsnet_arxiv_link])
- Official Code ([GitHub][efficient_capsnet_github_link])

[efficient_capsnet_arxiv_link]: https://arxiv.org/abs/2101.12491
[efficient_capsnet_github_link]: https://github.com/EscVM/Efficient-CapsNet