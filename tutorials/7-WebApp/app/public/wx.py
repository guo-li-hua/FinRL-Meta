# # -*- coding:utf-8 --*--
# from Cryptodome.PublicKey import RSA
# from Cryptodome.Cipher import PKCS1_v1_5
# from Cryptodome.Hash import SHA
# from Cryptodome import Random
# from base64 import b64decode,b64encode
#
# def generate_key():
#     """
#     生成公钥和私钥
#     """
#     # 返回私钥和公钥
#     rsa = RSA.generate(1024)
#     private_key = rsa.exportKey()
#     publick_key = rsa.publickey().exportKey()
#     print(publick_key.decode())
#     return private_key.decode(), publick_key.decode()
#
#
# def rsa_encrypt(public_key, message):
#     """
#     rsa加密函数
#     """
#     # publick_key: 公钥
#     # message: 需要加密的信息
#     # :return: 加密后的密文
#     public_key = RSA.import_key(public_key)
#     cipher_rsa = PKCS1_v1_5.new(public_key)
#     encrypt_text = []
#     for i in range(0, len(message), 100):
#         cont = message[i:i + 100]
#         encrypt_text.append(cipher_rsa.encrypt(cont.encode()))
#     # 加密完进行拼接
#     cipher_text = b''.join(encrypt_text)
#     # base64进行编码
#     result = b64encode(cipher_text)
#     return result.decode()
#
#
# def rsa_decrypt(private_key, message):
#     """
#     rsa解密函数
#     """
#     # private_key: 私钥
#     # message: 加密后的密文
#     # :return: 解密后原始信息
#     dsize = SHA.digest_size
#     sentinel = Random.new().read(1024 + dsize)
#     private_key = RSA.import_key(private_key)
#     cipher_rsa = PKCS1_v1_5.new(private_key)
#     # if (len(message) % 3 == 1):
#     #     message += "=="
#     # elif (len(message) % 3 == 2):
#     #     message += "="
#     # message = bytes(message, encoding='utf8')
#     data = b64encode(message.encode())
#     return cipher_rsa.decrypt(data, sentinel)