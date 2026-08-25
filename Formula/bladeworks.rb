# frozen_string_literal: true

# Architecture map:
#   Homebrew native libraries + pinned Apple Silicon Python wheels
#       -> isolated Python 3.12 virtual environment
#       -> Bladeworks installed without network dependency resolution
#       -> formula tests verify ffprobe, Pillow, and RAQM before release
class Bladeworks < Formula
  include Language::Python::Virtualenv

  desc "Render portable Final Cut Pro XML projects to video"
  homepage "https://github.com/bladeworks-ai/bladeworks"
  url "https://files.pythonhosted.org/packages/07/ec/9c392032b46bdcc071495d1bf0612f795ed423ac1cc37528c3a6217016ab/bladeworks-0.1.4.tar.gz"
  sha256 "d43812f3847b92dda19de7a53f0f224db522375562d3bf7235bf61ea7d69e882"
  license "AGPL-3.0-only"

  depends_on arch: :arm64
  depends_on "ffmpeg"
  depends_on "fribidi"
  depends_on "harfbuzz"
  depends_on "libraqm"
  depends_on macos: :sonoma
  depends_on "python@3.12"

  resource "torch" do
    url "https://files.pythonhosted.org/packages/6f/8b/69e3008d78e5cee2b30183340cc425081b78afc5eff3d080daab0adda9aa/torch-2.11.0-cp312-cp312-macosx_11_0_arm64.whl"
    sha256 "4b5866312ee6e52ea625cd211dcb97d6a2cdc1131a5f15cc0d87eec948f6dd34"
  end

  resource "av" do
    url "https://files.pythonhosted.org/packages/27/19/3a4d3882852a0ee136121979ce46f6d2867b974eb217a2c9a070939f55ad/av-16.0.1-cp312-cp312-macosx_14_0_arm64.whl"
    sha256 "6352a64b25c9f985d4f279c2902db9a92424e6f2c972161e67119616f0796cb9"
  end

  resource "numpy" do
    url "https://files.pythonhosted.org/packages/3c/65/4baa99f1c53b30adf0acd9a5519078871ddde8d2339dc5a7fde80d9d87da/numpy-2.2.6-cp312-cp312-macosx_14_0_arm64.whl"
    sha256 "894b3a42502226a1cac872f840030665f33326fc3dac8e57c607905773cdcde3"
  end

  resource "tqdm" do
    url "https://files.pythonhosted.org/packages/d0/30/dc54f88dd4a2b5dc8a0279bdd7270e735851848b762aeb1c1184ed1f6b14/tqdm-4.67.1-py3-none-any.whl"
    sha256 "26445eca388f82e72884e0d580d5464cd801a3ea01e63e5601bdff9ba6a48de2"
  end

  resource "pillow" do
    url "https://files.pythonhosted.org/packages/d8/95/0a351b9289c2b5cbde0bacd4a83ebc44023e835490a727b2a3bd60ddc0f4/pillow-12.2.0-cp312-cp312-macosx_11_0_arm64.whl"
    sha256 "f3f40b3c5a968281fd507d519e444c35f0ff171237f4fdde090dd60699458421"
  end

  resource "fonttools" do
    url "https://files.pythonhosted.org/packages/ba/3d/1f45db2df51e7bfa55492e8f23f383d372200be3a0ded4bf56a92753dd1f/fonttools-4.59.2-cp312-cp312-macosx_10_13_universal2.whl"
    sha256 "82906d002c349cad647a7634b004825a7335f8159d0d035ae89253b4abf6f3ea"
  end

  resource "setuptools" do
    url "https://files.pythonhosted.org/packages/e1/e3/c164c88b2e5ce7b24d667b9bd83589cf4f3520d97cad01534cd3c4f55fdb/setuptools-81.0.0-py3-none-any.whl"
    sha256 "fdd925d5c5d9f62e4b74b30d6dd7828ce236fd6ed998a08d81de62ce5a6310d6"
  end

  resource "fastapi" do
    url "https://files.pythonhosted.org/packages/cb/03/10388a42375ee7e4ac9b94eb2c5c569c8b5795e377e701c9ac3ad63de890/fastapi-0.141.1-py3-none-any.whl"
    sha256 "bfb91aa2d334c61cb35ba9a116fc123b3d3df31640b801cf57a7a78ec3f603b3"
  end

  resource "uvicorn" do
    url "https://files.pythonhosted.org/packages/f1/79/4a20b54ab0491485ccd8c077db2d39187c7f12b3e15485d38a7be37c81b4/uvicorn-0.52.4-py3-none-any.whl"
    sha256 "f86e41a149d7d05a9969337e3946a9c171c06a5d42680896daaba624aeac8da1"
  end

  resource "aiortc" do
    url "https://files.pythonhosted.org/packages/0c/5f/8435ba02c9278b6cec6f168db92e1d3280dd3af8f2225e20dc7c3be5ab22/aiortc-1.15.0-py3-none-any.whl"
    sha256 "4e1e54bff31a9c2cb654c7b7edc068085a7df53365e5df24a5cb24168e3f95f7"
  end

  resource "aioice" do
    url "https://files.pythonhosted.org/packages/c7/e3/0d23b1f930c17d371ce1ec36ee529f22fd19ebc2a07fe3418e3d1d884ce2/aioice-0.10.2-py3-none-any.whl"
    sha256 "14911c15ab12d096dd14d372ebb4aecbb7420b52c9b76fdfcf54375dec17fcbf"
  end

  resource "annotated-doc" do
    url "https://files.pythonhosted.org/packages/3e/30/e900b21425a860e195f32e37657aa1f7c7f2b1bfb26f03ca209b90933c06/annotated_doc-0.0.5-py3-none-any.whl"
    sha256 "117bac03a25ede5df5440e855b32d556049ca169ead221505badf432fed4b101"
  end

  resource "click" do
    url "https://files.pythonhosted.org/packages/fb/e2/79c688af8b210d232694e31e59da9f6ec747bae31c3f5946e4e9b98860d5/click-8.4.2-py3-none-any.whl"
    sha256 "e6f9f66136c816745b9d65817da91d61d957fb16e02e4dcd0552553c5a197b76"
  end

  resource "cryptography" do
    url "https://files.pythonhosted.org/packages/c5/5c/59086b4aac5e879d38ddbcf74e4be7ade89cebc3eb199a55da998c3bb46a/cryptography-50.0.0-cp311-abi3-macosx_11_0_arm64.whl"
    sha256 "031e2d5dd4bb9caa3ca9c82e5a197fd8ae680232cee62603d1a813f3f07e3d03"
  end

  resource "cffi" do
    url "https://files.pythonhosted.org/packages/54/7d/16e5a096677b5e313ca80cd5e5170efa3ea44624a82bb111925522da64b1/cffi-2.1.1-cp312-cp312-macosx_11_0_arm64.whl"
    sha256 "f81b3b8f3d4e343550fa4baa0e479bba9f2d29ce9c2e9b51d1ce1718d7442fcf"
  end

  resource "dnspython" do
    url "https://files.pythonhosted.org/packages/ba/5a/18ad964b0086c6e62e2e7500f7edc89e3faa45033c71c1893d34eed2b2de/dnspython-2.8.0-py3-none-any.whl"
    sha256 "01d9bbc4a2d76bf0db7c1f729812ded6d912bd318d3b1cf81d30c0f845dbf3af"
  end

  resource "fsspec" do
    url "https://files.pythonhosted.org/packages/fd/3c/6a2bf344106328fd04963664a60b9bb6496fc25df8e962fcdc1367285fb9/fsspec-2026.7.0-py3-none-any.whl"
    sha256 "b57ddbafedfaef7018c1ecab32aa200a9d7ca26b77965f64e48b70061249d279"
  end

  resource "google-crc32c" do
    url "https://files.pythonhosted.org/packages/e9/5f/7307325b1198b59324c0fa9807cafb551afb65e831699f2ce211ad5c8240/google_crc32c-1.8.0-cp312-cp312-macosx_12_0_arm64.whl"
    sha256 "4b8286b659c1335172e39563ab0a768b8015e88e08329fa5321f774275fc3113"
  end

  resource "h11" do
    url "https://files.pythonhosted.org/packages/04/4b/29cac41a4d98d144bf5f6d33995617b185d14b22401f75ca86f384e87ff1/h11-0.16.0-py3-none-any.whl"
    sha256 "63cf8bbe7522de3bf65932fda1d9c2772064ffb3dae62d55932da54b31cb6c86"
  end

  resource "ifaddr" do
    url "https://files.pythonhosted.org/packages/9c/1f/19ebc343cc71a7ffa78f17018535adc5cbdd87afb31d7c34874680148b32/ifaddr-0.2.0-py3-none-any.whl"
    sha256 "085e0305cfe6f16ab12d72e2024030f5d52674afad6911bb1eee207177b8a748"
  end

  resource "networkx" do
    url "https://files.pythonhosted.org/packages/9e/c9/b2622292ea83fbb4ec318f5b9ab867d0a28ab43c5717bb85b0a5f6b3b0a4/networkx-3.6.1-py3-none-any.whl"
    sha256 "d47fbf302e7d9cbbb9e2555a0d267983d2aa476bac30e90dfbe5669bd57f3762"
  end

  resource "pydantic" do
    url "https://files.pythonhosted.org/packages/fd/7b/122376b1fd3c62c1ed9dc80c931ace4844b3c55407b6fb2d199377c9736f/pydantic-2.13.4-py3-none-any.whl"
    sha256 "45a282cde31d808236fd7ea9d919b128653c8b38b393d1c4ab335c62924d9aba"
  end

  resource "pydantic-core" do
    url "https://files.pythonhosted.org/packages/19/95/6195171e385007300f0f5574592e467c568becce2d937a0b6804f218bc49/pydantic_core-2.46.4-cp312-cp312-macosx_11_0_arm64.whl"
    sha256 "962ccbab7b642487b1d8b7df90ef677e03134cf1fd8880bf698649b22a69371f"
  end

  resource "annotated-types" do
    url "https://files.pythonhosted.org/packages/99/91/8acff4f5e50511b911bbccb72b8628a49c68ce14148cd9f6431094859a90/annotated_types-0.8.0-py3-none-any.whl"
    sha256 "f072f4d804ea359e4eaf198b1af7a8b0943881a87f31bb764f8bf219bb9419e0"
  end

  resource "pyee" do
    url "https://files.pythonhosted.org/packages/81/12/5347938b1f9a6453f0dbdfcc3e2388a1320ef9b9ec17fbefbc4ab647ea98/pyee-14.0.0-py3-none-any.whl"
    sha256 "3ac2d3229a9677f7de2c33d7f52fe25b638a46b19c413fea2edc8c6d0a644e4d"
  end

  resource "pylibsrtp" do
    url "https://files.pythonhosted.org/packages/8d/0e/8d215484a9877adcf2459a8b28165fc89668b034565277fd55d666edd247/pylibsrtp-1.0.0-cp310-abi3-macosx_11_0_arm64.whl"
    sha256 "aaad74e5c8cbc1c32056c3767fea494c1e62b3aea2c908eda2a1051389fdad76"
  end

  resource "pyopenssl" do
    url "https://files.pythonhosted.org/packages/51/ad/2cf6d3fa2fae5c79e1ed9960c0d42badd0f94d81dd12b50604cdc839e648/pyopenssl-26.4.0-py3-none-any.whl"
    sha256 "f0eb0cb2d581d3ad2b9c489468485e7f2ab6727d08401bcf9d824c3caddf3c1c"
  end

  resource "starlette" do
    url "https://files.pythonhosted.org/packages/c8/cb/6a6a47d5b464bd08695d254f3da6e7986cc70c9fa5d778eda57538edfe56/starlette-1.6.0-py3-none-any.whl"
    sha256 "a86dd39d14bb45f85a3d18525215a9ef0cfd1f192ac793220e72598c90335f0c"
  end

  resource "anyio" do
    url "https://files.pythonhosted.org/packages/da/35/f2287558c17e29fafc8ef3daf819bb9834061cfa43bff8014f7df7f63bdc/anyio-4.14.2-py3-none-any.whl"
    sha256 "9f505dda5ac9f0c8309b5e8bd445a8c2bf7246f3ce950121e45ea15bc41d1494"
  end

  resource "idna" do
    url "https://files.pythonhosted.org/packages/57/b0/0e52c878c53f245edd3a11020f20979b3f490f245af532c7cae3027754b5/idna-3.19-py3-none-any.whl"
    sha256 "815e7be7a7806d54abb586dc943addc79e8b2ee16915059658cbeff4b1b43bf4"
  end

  resource "sympy" do
    url "https://files.pythonhosted.org/packages/a2/09/77d55d46fd61b4a135c444fc97158ef34a095e5681d0a6c10b75bf356191/sympy-1.14.0-py3-none-any.whl"
    sha256 "e091cc3e99d2141a0ba2847328f5479b05d94a6635cb96148ccb3f34671bd8f5"
  end

  resource "mpmath" do
    url "https://files.pythonhosted.org/packages/43/e3/7d92a15f894aa0c9c4b49b8ee9ac9850d6e63b03c9c32c0367a13ae62209/mpmath-1.3.0-py3-none-any.whl"
    sha256 "a0b2b9fe80bbcd81a6647ff13108738cfb482d481d826cc0e02f5b35e5c88d2c"
  end

  resource "typing-extensions" do
    url "https://files.pythonhosted.org/packages/49/d3/b8441a820a491ddfc024b0b0cf0393375b75ea13866d9c66727e54c2fc80/typing_extensions-4.16.0-py3-none-any.whl"
    sha256 "481caa481374e813c1b176ada14e97f1f67a4539ce9cfeb3f350d78d6370c2e8"
  end

  resource "typing-inspection" do
    url "https://files.pythonhosted.org/packages/67/81/4add07e5172b7ac40d8ed5ff580409a7801a4fe26d529bdd915401dabfbe/typing_inspection-0.4.4-py3-none-any.whl"
    sha256 "65b8397ba37ccbce054456aaccddfc91e6e3083c92824df348d96ca832f3f147"
  end

  resource "filelock" do
    url "https://files.pythonhosted.org/packages/01/a4/9b63d595d748e3aff8812b65eacc1a2c4bd90b7c2012e08e72373b4835eb/filelock-3.32.4-py3-none-any.whl"
    sha256 "22e58ca3b1ae3b98993b762d7338367ae64fe50252bf78d59da3bfebcdf1cedd"
  end

  resource "jinja2" do
    url "https://files.pythonhosted.org/packages/62/a1/3d680cbfd5f4b8f15abc1d571870c5fc3e594bb582bc3b64ea099db13e56/jinja2-3.1.6-py3-none-any.whl"
    sha256 "85ece4451f492d0c13c5dd7c13a64681a86afae63a5f347908daf103ce6d2f67"
  end

  resource "markupsafe" do
    url "https://files.pythonhosted.org/packages/9a/81/7e4e08678a1f98521201c3079f77db69fb552acd56067661f8c2f534a718/markupsafe-3.0.3-cp312-cp312-macosx_11_0_arm64.whl"
    sha256 "1872df69a4de6aead3491198eaf13810b565bdbeec3ae2dc8780f14458ec73ce"
  end

  resource "pycparser" do
    url "https://files.pythonhosted.org/packages/0c/c3/44f3fbbfa403ea2a7c779186dc20772604442dde72947e7d01069cbe98e3/pycparser-3.0-py3-none-any.whl"
    sha256 "b727414169a36b7d524c1c3e31839a521725078d7b2ff038656844266160a992"
  end

  resource "wheel" do
    url "https://files.pythonhosted.org/packages/2e/29/69cfbb602cd91690c55d38ba9fe53e6a7e76a6fa647bf38f19c138d25449/wheel-0.48.0-py3-none-any.whl"
    sha256 "3217dcc807155e45db462d7ef2431f5ddda0d7273b700d05a67b271ceb1287ab"
  end

  def install
    venv = virtualenv_create(libexec, "python3.12")
    resources.each do |resource|
      resource.stage do
        wheel = Pathname.pwd/resource.downloader.basename
        venv.pip_install wheel
      end
    end
    venv.pip_install_and_link buildpath, build_isolation: false
  end

  test do
    output = shell_output("#{bin}/bladeworks doctor")
    assert_match "ffprobe: OK (#{formula_opt_bin("ffmpeg")}/ffprobe)", output
    assert_equal "12.2.0", shell_output("#{libexec}/bin/python -c 'import PIL; print(PIL.__version__)'").strip
    system libexec/"bin/python", "-c", 'from PIL import features; assert features.check("raqm")'
  end
end
