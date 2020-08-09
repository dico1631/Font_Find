from imgurpython import ImgurClient

class imguruploader:
    def __init__(self):
        self.client = self.authenticate()
        self.config = self.setConfig(None, '', '', '')
        self.image_path = ''

    def authenticate(self) :
        # If you already have an access/refresh pair in hand
        client_id = '5966e128f842fb7'
        client_secret = 'f2d8f4656c36dff8fef41e531ec377c90886c355'
        access_token = 'ce1aa78d9b7733f6bd5260a6b1e5056b7aa8cc95'
        refresh_token = '3d3b96dd1c9e3b0fd769dc0a123ca8667a0a2ed1'

        # Note since access tokens expire after an hour, only the refresh token is required (library handles autorefresh)
        self.client = ImgurClient(client_id, client_secret, access_token, refresh_token)
        
        authorization_url = self.client.get_auth_url('pin')
        self.client.set_user_auth(access_token, refresh_token)
        print("Authentication successful")

        return self.client

    def setConfig(self, album, name, title, description):
        self.config = {
            'album' : album,
            'name' : name,
            'title' : title,
            'description' : description
        }
        return self.config

    def setpath(self, image_path):
        self.image_path = image_path

    def imgurUpload(self):
        print('Uploading Image....-----------------------------------------------------------------------')
        image = self.client.upload_from_path(self.image_path, config = self.config, anon = False)
        print("Done")
        print()
        return image