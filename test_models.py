"""
Test your Hugging Face model setup
Run this to verify your installation is working
"""
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from sentence_transformers import SentenceTransformer
from config import Config

def test_device():
    """Test available compute devices"""
    print("="*60)
    print("🔍 TESTING COMPUTE DEVICES")
    print("="*60)
    
    print(f"\nCUDA Available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA Device: {torch.cuda.get_device_name(0)}")
        print(f"CUDA Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    
    print(f"MPS Available (Apple Silicon): {torch.backends.mps.is_available() if hasattr(torch.backends, 'mps') else False}")
    print(f"CPU Cores: {torch.get_num_threads()}")
    
    print(f"\nConfigured Device: {Config.DEVICE}")

def test_embeddings():
    """Test embeddings model"""
    print("\n" + "="*60)
    print("🔍 TESTING EMBEDDINGS MODEL")
    print("="*60)
    
    try:
        print(f"\nLoading: {Config.EMBEDDING_MODEL}")
        model = SentenceTransformer(Config.EMBEDDING_MODEL)
        
        # Test embedding
        test_text = "This is a test sentence."
        embedding = model.encode(test_text)
        
        print(f"✓ Model loaded successfully!")
        print(f"  Embedding dimension: {len(embedding)}")
        print(f"  Device: {model.device}")
        
        return True
    except Exception as e:
        print(f"❌ Error loading embeddings: {e}")
        return False

def test_llm():
    """Test LLM model"""
    print("\n" + "="*60)
    print("🔍 TESTING LLM")
    print("="*60)
    
    try:
        print(f"\nLoading: {Config.LLM_MODEL}")
        print("This may take a few minutes on first run...")
        
        tokenizer = AutoTokenizer.from_pretrained(Config.LLM_MODEL)
        
        # Load model with appropriate settings
        if Config.DEVICE == "cuda":
            print("Loading with 8-bit quantization for GPU...")
            model = AutoModelForCausalLM.from_pretrained(
                Config.LLM_MODEL,
                torch_dtype=torch.float16,
                device_map="auto",
                load_in_8bit=True
            )
        else:
            print("Loading for CPU (this may take longer)...")
            model = AutoModelForCausalLM.from_pretrained(
                Config.LLM_MODEL,
                torch_dtype=torch.float32,
            )
        
        print(f"✓ Model loaded successfully!")
        print(f"  Device: {next(model.parameters()).device}")
        print(f"  Dtype: {next(model.parameters()).dtype}")
        print(f"  Parameters: ~{sum(p.numel() for p in model.parameters()) / 1e9:.1f}B")
        
        # Test generation
        print("\nTesting generation...")
        test_prompt = "### Question: What is AI?\n### Answer:"
        inputs = tokenizer(test_prompt, return_tensors="pt")
        
        if Config.DEVICE == "cuda":
            inputs = {k: v.cuda() for k, v in inputs.items()}
        
        outputs = model.generate(
            **inputs,
            max_new_tokens=50,
            temperature=0.7,
            do_sample=True
        )
        
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print(f"\nTest Response:\n{response}")
        
        return True
    except Exception as e:
        print(f"❌ Error loading LLM: {e}")
        print("\nTroubleshooting:")
        print("- If out of memory: try DEVICE=cpu in .env")
        print("- If model not found: check model name in .env")
        print("- See HUGGINGFACE_GUIDE.md for alternatives")
        return False

def main():
    print("""
    ╔════════════════════════════════════════════════════════╗
    ║   HUGGING FACE MODEL TEST                              ║
    ╚════════════════════════════════════════════════════════╝
    """)
    
    # Test device
    test_device()
    
    # Test embeddings (small, fast)
    embeddings_ok = test_embeddings()
    
    # Test LLM (large, may take time)
    if embeddings_ok:
        print("\n" + "!"*60)
        print("⚠ WARNING: LLM test will download ~14GB on first run")
        print("!"*60)
        response = input("\nContinue with LLM test? (y/n): ").strip().lower()
        
        if response == 'y':
            llm_ok = test_llm()
        else:
            print("\nSkipping LLM test. Run again when ready.")
            llm_ok = None
    else:
        print("\n❌ Embeddings test failed. Fix this before testing LLM.")
        llm_ok = False
    
    # Summary
    print("\n" + "="*60)
    print("📊 TEST SUMMARY")
    print("="*60)
    print(f"Embeddings: {'✓ PASS' if embeddings_ok else '❌ FAIL'}")
    if llm_ok is not None:
        print(f"LLM: {'✓ PASS' if llm_ok else '❌ FAIL'}")
    else:
        print("LLM: ⊘ SKIPPED")
    
    if embeddings_ok and (llm_ok or llm_ok is None):
        print("\n🎉 Setup is ready! You can now run:")
        print("   python main.py 'Your Research Topic'")
    else:
        print("\n⚠ Please fix errors above before using the system")
        print("   See HUGGINGFACE_GUIDE.md for troubleshooting")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n👋 Test interrupted by user")
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
