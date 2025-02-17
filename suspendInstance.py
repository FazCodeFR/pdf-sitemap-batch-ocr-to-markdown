import os
import openstack

# Charger les variables d'environnement depuis le fichier openrc.sh
auth_url = os.getenv('OS_AUTH_URL')
project_name = os.getenv('OS_PROJECT_NAME')
username = os.getenv('OS_USERNAME')
password = os.getenv('OS_PASSWORD')
user_domain_name = os.getenv('OS_USER_DOMAIN_NAME')
project_domain_name = os.getenv('OS_PROJECT_DOMAIN_NAME')
region_name = os.getenv('OS_REGION_NAME')

# Connexion à OpenStack
conn = openstack.connect(
    auth_url=auth_url,
    project_name=project_name,
    username=username,
    password=password,
    region_name=region_name,
    user_domain_name=user_domain_name,
    project_domain_name=project_domain_name
)

# Obtenir la liste des instances
instances = list(conn.compute.servers())

# Vérifier s'il y a des instances
if instances:
    # Vérifier si l'instance est déjà en train de se shelver
    first_instance = instances[0]
    
    if first_instance.task_state == 'shelving':
        print(f"Instance '{first_instance.name}' avec l'ID '{first_instance.id}' est déjà en cours de shelve.")
    elif first_instance.status == 'SHELVED':
        print(f"Instance '{first_instance.name}' avec l'ID '{first_instance.id}' est déjà shelvée.")
    else:
        # Shelver l'instance
        conn.compute.shelve_server(first_instance.id)
        print(f"Instance '{first_instance.name}' avec l'ID '{first_instance.id}' a été shelvée.")
else:
    print("Aucune instance trouvée.")
